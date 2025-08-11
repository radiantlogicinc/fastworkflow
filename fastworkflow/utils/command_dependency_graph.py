import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from sentence_transformers import SentenceTransformer, util as st_util

from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.utils import python_utils


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


def _serialize_type_str(t: Any) -> str:
    try:
        # Basic serialization for common typing and classes
        return t.__name__ if hasattr(t, "__name__") else str(t)
    except Exception:
        return str(t)


def _collect_command_params(workflow_path: str) -> Dict[str, CommandParams]:
    directory = CommandDirectory.load(workflow_path)
    routing = RoutingDefinition.build(workflow_path)

    results: Dict[str, CommandParams] = {}

    for qualified_name in directory.get_commands():
        # Hydrate metadata and import module
        directory.ensure_command_hydrated(qualified_name)
        metadata = directory.get_command_metadata(qualified_name)
        module_path = metadata.parameter_extraction_signature_module_path or metadata.response_generation_module_path
        if not module_path:
            continue
        module = python_utils.get_module(str(module_path), workflow_path)
        signature_cls = getattr(module, "Signature", None)
        if not signature_cls:
            continue

        inputs: List[ParamMeta] = []
        outputs: List[ParamMeta] = []

        InputModel = getattr(signature_cls, "Input", None)
        if InputModel is not None and hasattr(InputModel, "model_fields"):
            for name, field in InputModel.model_fields.items():
                desc = getattr(field, "description", "") or ""
                examples = getattr(field, "examples", []) or []
                type_str = _serialize_type_str(field.annotation)
                inputs.append(ParamMeta(name=name, type_str=type_str, description=str(desc), examples=list(examples)))

        OutputModel = getattr(signature_cls, "Output", None)
        if OutputModel is not None and hasattr(OutputModel, "model_fields"):
            for name, field in OutputModel.model_fields.items():
                desc = getattr(field, "description", "") or ""
                examples = getattr(field, "examples", []) or []
                type_str = _serialize_type_str(field.annotation)
                outputs.append(ParamMeta(name=name, type_str=type_str, description=str(desc), examples=list(examples)))

        if contexts := routing.get_contexts_for_command(qualified_name):
            results[qualified_name] = CommandParams(inputs=inputs, outputs=outputs)

    return results


def _exact_match_score(out_param: ParamMeta, in_param: ParamMeta) -> float:
    if out_param.name.lower() == in_param.name.lower() and out_param.type_str == in_param.type_str:
        return 1.0
    return 0.0


def _semantic_match_score(out_param: ParamMeta, in_param: ParamMeta) -> float:
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
        return 0.0

    # Lazy-load model and cache embeddings to avoid repeated work
    global _st_model  # type: ignore
    global _embedding_cache  # type: ignore
    try:
        _st_model
    except NameError:
        _st_model = None  # type: ignore
    try:
        _embedding_cache
    except NameError:
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
    return float(sim.item())


def _contexts_overlap(routing: RoutingDefinition, cmd_x: str, cmd_y: str) -> bool:
    cx = routing.get_contexts_for_command(cmd_x)
    cy = routing.get_contexts_for_command(cmd_y)
    return False if not cx or not cy else bool(cx & cy)


def generate_dependency_graph(workflow_path: str, semantic_threshold: float = 0.85, exact_only: bool = False) -> str:
    """
    Build the parameter dependency graph and persist as JSON in ___command_info/parameter_dependency_graph.json.

    Returns the path to the generated JSON file.
    """
    params_by_command = _collect_command_params(workflow_path)
    routing = RoutingDefinition.build(workflow_path)

    nodes = sorted(list(params_by_command.keys()))
    edges: List[Dict[str, Any]] = []

    for y in nodes:
        for x in nodes:
            if x == y:
                continue
            if not _contexts_overlap(routing, x, y):
                continue

            x_outputs = params_by_command[x].outputs
            y_inputs = params_by_command[y].inputs
            if not x_outputs or not y_inputs:
                continue

            matched: List[Tuple[str, str, float, str]] = []  # (y_input, x_output, score, match_type)
            for xo in x_outputs:
                for yi in y_inputs:
                    score = _exact_match_score(xo, yi)
                    match_type = "exact" if score == 1.0 else "semantic"
                    if score < 1.0 and not exact_only:
                        score = _semantic_match_score(xo, yi)
                    if score >= (1.0 if match_type == "exact" else semantic_threshold):
                        matched.append((yi.name, xo.name, score, match_type))

            if matched:
                # Aggregate edge weight as average of scores
                weight = sum(m[2] for m in matched) / len(matched)
                match_types = {m[3] for m in matched}
                edge = {
                    "from": y,
                    "to": x,
                    "weight": float(weight),
                    "matched_params": [(m[0], m[1]) for m in matched],
                    "match_type": "exact" if match_types == {"exact"} else ("semantic" if match_types == {"semantic"} else "mixed"),
                }
                edges.append(edge)

    artifact_dir = CommandDirectory.get_commandinfo_folderpath(workflow_path)
    out_path = os.path.join(artifact_dir, "parameter_dependency_graph.json")
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
    min_weight: float = 0.7,
    max_depth: int = 3,
) -> List[Dict[str, Any]]:
    """
    Recursively resolves dependencies, returning a list of plans for resolving the missing param.
    """
    graph = _load_graph(graph_path)
    edges = graph.get("edges", [])

    # Build adjacency: from -> list of (to, edge_data)
    adj: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        if e.get("weight", 0.0) < min_weight:
            continue
        adj.setdefault(e["from"], []).append(e)

    def recurse(node: str, depth: int) -> List[Dict[str, Any]]:
        if depth > max_depth:
            return []
        plans: List[Dict[str, Any]] = []
        for e in adj.get(node, []):
            # Only consider edges where this missing param is matched
            matched_params = e.get("matched_params", [])
            if all(mp[0] != missing_input_param for mp in matched_params):
                continue
            neighbor = e["to"]
            sub_plans = recurse(neighbor, depth + 1)
            plans.append({
                "command": neighbor,
                "sub_plans": sub_plans,
                "weight": e.get("weight", 0.0),
            })
        return plans

    dependency_plans = recurse(y_qualified_name, 0)
    # Prefer shallower trees, then higher weights
    dependency_plans.sort(key=lambda p: (len(p.get("sub_plans", [])), -p.get("weight", 0.0)))
    return dependency_plans
