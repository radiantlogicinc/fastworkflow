import os
import json
import ast
import glob
from typing import Any, Dict, List, Optional, Tuple

from fastworkflow.utils.logging import logger

# The post-processor is intentionally conservative and disabled by default to avoid
# affecting existing build/test flows unless explicitly enabled.
# Enable by setting FASTWORKFLOW_ENABLE_DSPY_POSTPROCESSING=1


def _is_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    env = env or os.environ
    return env.get("FASTWORKFLOW_ENABLE_DSPY_POSTPROCESSING", "0") in {"1", "true", "True"}


def _safe_import_dspy() -> Optional[Any]:
    try:
        import dspy  # type: ignore

        return dspy
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(f"DSPy not available or failed to import: {exc}")
        return None


def _list_command_files(commands_root: str) -> List[str]:
    return [
        f
        for f in glob.glob(os.path.join(commands_root, "**", "*.py"), recursive=True)
        if not f.endswith("__init__.py") and not os.path.basename(f).startswith("_")
    ]


def _load_context_model(commands_root: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    context_model_path = os.path.join(commands_root, "context_inheritance_model.json")
    if not os.path.isfile(context_model_path):
        return None, f"Context model not found at {context_model_path}"
    try:
        with open(context_model_path, "r") as fh:
            return json.load(fh), None
    except Exception as exc:  # pragma: no cover - defensive fallback
        return None, f"Failed to parse context model JSON: {exc}"


def _parse_python_ast(filepath: str) -> Optional[ast.AST]:
    try:
        with open(filepath, "r") as fh:
            return ast.parse(fh.read())
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(f"AST parse failed for {filepath}: {exc}")
        return None


def _extract_signature_info(tree: ast.AST) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "has_signature": False,
        "input_fields": [],
        "output_fields": [],
        "plain_utterances": [],
        "docstring": None,
    }
    try:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Signature":
                info["has_signature"] = True
                # Collect plain_utterances
                for body_item in node.body:
                    if isinstance(body_item, ast.Assign):
                        for target in body_item.targets:
                            if isinstance(target, ast.Name) and target.id == "plain_utterances":
                                try:
                                    if isinstance(body_item.value, (ast.List, ast.Tuple)):
                                        values = []
                                        for elt in body_item.value.elts:
                                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                                values.append(elt.value)
                                        info["plain_utterances"] = values
                                except Exception:  # pragma: no cover
                                    pass
                break
    except Exception:  # pragma: no cover - defensive
        pass
    return info


def _write_workflow_description(workflow_root: str, description: str) -> None:
    out_path = os.path.join(workflow_root, "workflow_description.txt")
    try:
        with open(out_path, "w") as fh:
            fh.write(description)
        logger.info(f"Generated workflow description at {out_path}")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to write workflow_description.txt: {exc}")


def _generate_description_with_dspy(dspy_mod: Any, contexts: List[Dict[str, Any]], global_cmds: List[Dict[str, str]]) -> str:
    # Minimal inline program per spec; fallback to a succinct summary if anything fails
    try:
        class WorkflowDescription(dspy_mod.Signature):  # type: ignore[attr-defined]
            """Generate overall workflow description."""
            contexts: List[Dict[str, Any]]
            global_commands: List[Dict[str, str]]
            description: str = dspy_mod.OutputField(desc="High-level workflow overview.")

        program = dspy_mod.ChainOfThought(WorkflowDescription)  # type: ignore[attr-defined]
        result = program(contexts=contexts, global_commands=global_cmds)
        if getattr(result, "description", None):
            return result.description
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(f"DSPy workflow description generation failed: {exc}")
    # Fallback summary
    context_names = ", ".join(sorted({c.get("context_name", "?") for c in contexts}))
    global_names = ", ".join(sorted({g.get("name", "?") for g in global_cmds}))
    return f"Contexts: {context_names or 'none'}. Global commands: {global_names or 'none'}."


def _collect_for_description(commands_root: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    contexts: Dict[str, Dict[str, Any]] = {}
    global_cmds: List[Dict[str, str]] = []
    for fp in _list_command_files(commands_root):
        rel = os.path.relpath(fp, commands_root)
        parts = rel.split(os.sep)
        if len(parts) == 1:  # top-level command -> treat as global
            try:
                name = os.path.splitext(parts[0])[0]
                global_cmds.append({"name": name})
            except Exception:
                pass
            continue
        context = parts[0]
        cmd_name = os.path.splitext(parts[-1])[0]
        contexts.setdefault(context, {"context_name": context, "commands": []})
        contexts[context]["commands"].append({"name": cmd_name})
    return list(contexts.values()), global_cmds


def run_genai_postprocessor(args) -> None:
    """
    Execute the DSPy-based post-processing phase after deterministic generation.

    - Disabled by default unless FASTWORKFLOW_ENABLE_DSPY_POSTPROCESSING=1
    - On any failure, logs and returns without raising, per spec's fallback guidance
    - Currently implements:
      - Collection and optional generation of a workflow_description.txt via DSPy
      - Scaffolding for future in-place file enhancements (field metadata, utterances, docstrings)
    """
    try:
        if not hasattr(args, "workflow_folderpath"):
            logger.debug("Post-processor skipped: missing workflow_folderpath in args")
            return
        workflow_root = args.workflow_folderpath
        commands_root = os.path.join(workflow_root, "_commands")

        if not _is_enabled():
            logger.info("GenAI post-processing disabled. Set FASTWORKFLOW_ENABLE_DSPY_POSTPROCESSING=1 to enable.")
            return

        if not os.path.isdir(commands_root):
            logger.warning(f"Post-processor skipped: commands dir not found at {commands_root}")
            return

        # Load context model (optional, informative)
        context_model, err = _load_context_model(commands_root)
        if err:
            logger.warning(f"Post-processor: {err}")

        # Prepare DSPy
        dspy_mod = _safe_import_dspy()
        if dspy_mod is None:
            logger.info("DSPy unavailable; post-processing no-op.")
            return

        # 1) Prepare workflow description
        contexts, global_cmds = _collect_for_description(commands_root)
        if context_model and isinstance(context_model, dict):
            # Optionally enrich description inputs with docstrings later
            pass

        description_text = _generate_description_with_dspy(dspy_mod, contexts, global_cmds)
        _write_workflow_description(workflow_root, description_text)

        # 2) Placeholder: Iterate command files to enhance metadata/utterances/docstrings
        #    Intentionally conservative: do not mutate files yet without additional guardrails/tests
        for cmd_fp in _list_command_files(commands_root):
            tree = _parse_python_ast(cmd_fp)
            if tree is None:
                continue
            sig_info = _extract_signature_info(tree)
            # Here we would apply DSPy programs to:
            #  - Enrich Field(...) descriptions + examples/constraints via json_schema_extra
            #  - Generate minimal utterances to replace/augment plain_utterances
            #  - Add docstrings to Signature classes
            # For now, we log intent without modifying files to avoid disrupting deterministic tests.
            logger.debug(
                "[GenAI PP] Scanned %s | has_signature=%s | plain_utterances=%d",
                os.path.relpath(cmd_fp, commands_root),
                sig_info.get("has_signature"),
                len(sig_info.get("plain_utterances") or []),
            )

        logger.info("GenAI post-processing completed.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"GenAI post-processing encountered an error and was skipped: {exc}")