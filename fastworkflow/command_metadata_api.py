"""
This module provides common utilities for FastWorkflow, 
including a centralized API for extracting command metadata.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, List
import inspect
from pathlib import Path
import json

import fastworkflow
from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.utils import python_utils

def _is_pydantic_undefined(value: Any) -> bool:
    """Return True if value appears to be Pydantic's 'undefined' sentinel.

    We avoid importing Pydantic internals; instead detect by type name to keep
    compatibility across Pydantic versions.
    """
    try:
        type_name = type(value).__name__
        return type_name in {"PydanticUndefined", "PydanticUndefinedType"}
    except Exception:
        return False

class CommandMetadataAPI:
    """
    Provides a centralized API for extracting command metadata.
    """

    @staticmethod
    def get_enhanced_command_info(
        subject_workflow_path: str,
        cme_workflow_path: str,
        active_context_name: str,
    ) -> Dict[str, Any]:
        """
        Get enhanced command information for a given workflow and context.

        Args:
            subject_workflow_path: Path to the subject workflow
            cme_workflow_path: Path to the Command Metadata Extraction workflow
            active_context_name: Name of the active command context

        Returns:
            Structured dictionary with context and command details
        """
        subject_crd = fastworkflow.RoutingRegistry.get_definition(subject_workflow_path)
        cme_crd = fastworkflow.RoutingRegistry.get_definition(cme_workflow_path)

        cme_command_names = cme_crd.get_command_names('IntentDetection')
        subject_command_names = subject_crd.get_command_names(active_context_name)

        candidate_commands = set(cme_command_names) | set(subject_command_names)

        commands = []
        for fq_cmd in candidate_commands:
            if fq_cmd == "wildcard":
                continue

            utterance_meta = (
                subject_crd.command_directory.get_utterance_metadata(fq_cmd) or
                cme_crd.command_directory.get_utterance_metadata(fq_cmd)
            )

            if not utterance_meta:
                continue

            cmd_name = fq_cmd.split("/")[-1]
            signature_info = CommandMetadataAPI._extract_signature_info(fq_cmd, subject_crd, cme_crd)
            
            commands.append({
                "qualified_name": fq_cmd,
                "name": cmd_name,
                **signature_info
            })
        
        # This part is simplified as context info is now built outside
        return {"commands": sorted(commands, key=lambda x: x["name"])}

    @staticmethod
    def _extract_signature_info(fq_cmd: str, subject_crd: RoutingDefinition, cme_crd: RoutingDefinition) -> Dict[str, Any]:
        """
        Extracts signature information for a command.
        """
        signature_info = {}
        with contextlib.suppress(Exception):
            signature_class = None
            try:
                signature_class = subject_crd.get_command_class(
                    fq_cmd,
                    fastworkflow.ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS,
                )
            except Exception:
                with contextlib.suppress(Exception):
                    signature_class = cme_crd.get_command_class(
                        fq_cmd,
                        fastworkflow.ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS,
                    )

            if signature_class:
                sig_class = signature_class
                # Attach docstring from the Signature class itself
                with contextlib.suppress(Exception):
                    if doc := inspect.getdoc(sig_class) or getattr(
                        sig_class, "__doc__", None
                    ):
                        signature_info["doc_string"] = doc

                if hasattr(sig_class, 'Input') and hasattr(sig_class.Input, 'model_fields'):
                    input_class = sig_class.Input
                    inputs_list: List[Dict[str, Any]] = []
                    for field_name, field_info in input_class.model_fields.items():
                        pattern_value = CommandMetadataAPI._get_field_pattern(field_info)
                        # Extract optional 'available_from' from json_schema_extra
                        available_from_value = None
                        with contextlib.suppress(Exception):
                            extra = getattr(field_info, 'json_schema_extra', None)
                            if isinstance(extra, dict) and extra.get('available_from'):
                                available_from_value = str(extra.get('available_from'))
                        inputs_list.append(
                            {
                                "name": field_name,
                                "type": str(getattr(field_info, 'annotation', '')),
                                "description": getattr(field_info, 'description', "") or "",
                                "examples": list(getattr(field_info, 'examples', []) or []),
                                "default": (None if _is_pydantic_undefined(getattr(field_info, 'default', None)) else getattr(field_info, 'default', None)),
                                "pattern": pattern_value,
                                "available_from": available_from_value,
                            }
                        )
                    signature_info["inputs"] = inputs_list

                # Include output model fields if defined
                if hasattr(sig_class, 'Output') and hasattr(sig_class.Output, 'model_fields'):
                    output_class = sig_class.Output
                    outputs_list: List[Dict[str, Any]] = []
                    outputs_list.extend(
                        {
                            "name": field_name,
                            "type": str(getattr(field_info, 'annotation', '')),
                            "description": getattr(field_info, 'description', "")
                            or "",
                            "examples": list(
                                getattr(field_info, 'examples', []) or []
                            ),
                        }
                        for field_name, field_info in output_class.model_fields.items()
                    )
                    signature_info["outputs"] = outputs_list

                if hasattr(sig_class, 'plain_utterances'):
                    signature_info["plain_utterances"] = list(getattr(sig_class, 'plain_utterances') or [])
        return signature_info

    @staticmethod
    def _get_field_pattern(field_info: Any) -> Any:
        """
        Best-effort extraction of a pattern/regex constraint from a Pydantic field across versions.
        Returns a string pattern if available, otherwise None.
        """
        # Direct attributes commonly present across versions
        with contextlib.suppress(Exception):
            # pydantic v1 sometimes uses 'regex'
            regex_attr = getattr(field_info, 'regex', None)
            if regex_attr is not None:
                try:
                    return regex_attr.pattern if hasattr(regex_attr, 'pattern') else str(regex_attr)
                except Exception:
                    return str(regex_attr)

        with contextlib.suppress(Exception):
            if pattern_attr := getattr(field_info, 'pattern', None):
                return str(pattern_attr)

        with contextlib.suppress(Exception):
            extra = getattr(field_info, 'json_schema_extra', None)
            if isinstance(extra, dict) and 'pattern' in extra and extra['pattern']:
                return str(extra['pattern'])

        # Inspect metadata/annotation for Annotated[StringConstraints]
        with contextlib.suppress(Exception):
            annotation = getattr(field_info, 'annotation', None)
            # Attempt to unwrap typing.Annotated
            with contextlib.suppress(Exception):
                from typing import get_origin, get_args, Annotated  # type: ignore
                if get_origin(annotation) is Annotated:
                    for meta in get_args(annotation)[1:]:
                        # StringConstraints in pydantic v2
                        if hasattr(meta, 'pattern') and getattr(meta, 'pattern'):
                            return str(getattr(meta, 'pattern'))
                        if hasattr(meta, 'regex') and getattr(meta, 'regex'):
                            rx = getattr(meta, 'regex')
                            return rx.pattern if hasattr(rx, 'pattern') else str(rx)
        # Some versions store constraints in 'metadata' tuple
        with contextlib.suppress(Exception):
            metadata = getattr(field_info, 'metadata', None)
            if metadata and isinstance(metadata, (list, tuple)):
                for item in metadata:
                    if hasattr(item, 'pattern') and getattr(item, 'pattern'):
                        return str(getattr(item, 'pattern'))
                    if hasattr(item, 'regex') and getattr(item, 'regex'):
                        rx = getattr(item, 'regex')
                        return rx.pattern if hasattr(rx, 'pattern') else str(rx)
        return None

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _simplify_type_str(type_str: str) -> str:
        """Simplify verbose type strings like "<class 'float'>" to "float"."""
        if type_str.startswith("<class '") and type_str.endswith("'>"):
            return type_str[len("<class '"):-2]
        return type_str

    @staticmethod
    def _prune_empty(value: Any, remove_keys: set[str] | None = None) -> Any:
        """
        Recursively remove keys with empty values (None, '', [], {}) from dicts/lists.
        Optionally remove keys listed in remove_keys regardless of their values.
        """
        if remove_keys is None:
            remove_keys = set()
        if isinstance(value, dict):
            pruned: Dict[str, Any] = {}
            for k, v in value.items():
                if k in remove_keys:
                    continue
                pruned_v = CommandMetadataAPI._prune_empty(v, remove_keys)
                is_empty = (
                    pruned_v is None
                    or (isinstance(pruned_v, str) and pruned_v == "")
                    or (isinstance(pruned_v, (list, tuple, set)) and len(pruned_v) == 0)
                    or (isinstance(pruned_v, dict) and len(pruned_v) == 0)
                )
                # Preserve 'commands' key even if empty so headers still render
                if k == "commands":
                    is_empty = False
                if not is_empty:
                    pruned[k] = pruned_v
            return pruned
        if isinstance(value, list):
            new_list = [CommandMetadataAPI._prune_empty(v, remove_keys) for v in value]
            new_list = [v for v in new_list if not (
                v is None
                or (isinstance(v, str) and v == "")
                or (isinstance(v, (list, tuple, set)) and len(v) == 0)
                or (isinstance(v, dict) and len(v) == 0)
            )]
            return new_list
        return value

    @staticmethod
    def _to_yaml_like(value: Any, indent: int = 0, omit_command_name: bool = False) -> str:
        """Render a Python structure into a readable YAML-like string with custom formatting rules."""
        indent_str = "  " * indent
        lines: List[str] = []
        if isinstance(value, dict):
            for k, v in value.items():
                # 1) Omit the 'context' section entirely
                if k == "context":
                    continue
                # 2) Omit 'qualified_name' keys wherever they appear
                if k == "qualified_name":
                    continue
                # 3) Rename 'commands' header to 'Commands available'
                # display_key = "Commands available" if k == "commands" else k

                if isinstance(v, dict):
                    lines.extend(
                        (
                            # f"{indent_str}{display_key}:",
                            CommandMetadataAPI._to_yaml_like(v, indent + 1, omit_command_name=omit_command_name),
                        )
                    )
                elif isinstance(v, list):
                    # Special rendering for command entries: always show header even if empty
                    if k == "commands":
                        # lines.append(f"{indent_str}{display_key}:")
                        for idx, item in enumerate(v):
                            if isinstance(item, dict):
                                if cmd_name := item.get("name", ""):
                                    if not omit_command_name:
                                        lines.append(f"{indent_str}- {cmd_name}")
                                    if doc_text := item.get("doc_string"):
                                        if doc_text := " ".join(
                                            str(doc_text).split()
                                        ).strip():
                                            processed_cmd_name = cmd_name.replace('_', ' ').lower()
                                            if processed_cmd_name not in doc_text.lower() and doc_text.lower() != processed_cmd_name:
                                                if omit_command_name:
                                                    lines.append(f"{doc_text}")
                                                else:
                                                    lines.append(f"{indent_str}  {doc_text}")
                                elif not omit_command_name:
                                    lines.append(f"{indent_str}-")

                                # Render remaining fields (excluding name and qualified_name)
                                for rk, rv in item.items():
                                    if rk in {"name", "qualified_name", "doc_string", "plain_utterances"}:
                                        continue

                                    # 5) For inputs/outputs, render description first (fallback to name),
                                    # omit 'name' and 'description' keys, and avoid bare '-' lines
                                    if rk in {"inputs", "outputs"} and isinstance(rv, list):
                                        if not rv:
                                            continue
                                        lines.append(f"{indent_str}  {rk}:")
                                        for param in rv:
                                            if isinstance(param, dict):
                                                desc_val = str(param.get("description", "") or "").strip()
                                                name_val = str(param.get("name", "") or "").strip()
                                                title_val = desc_val or name_val
                                                type_val = str(param.get("type", "") or "").strip()
                                                if not title_val:
                                                    # Skip to avoid a bare '-'
                                                    continue
                                                # Render name/description and type on the same line
                                                if type_val:
                                                    lines.append(f"{indent_str}  - {title_val}, type: {type_val}")
                                                else:
                                                    lines.append(f"{indent_str}  - {title_val}")
                                                for rk2, rv2 in param.items():
                                                    # We've already rendered name/description (and type) inline
                                                    if rk2 in {"name", "description", "type"}:
                                                        continue
                                                    # Render examples inline for readability
                                                    if rk2 == "examples" and isinstance(rv2, list):
                                                        try:
                                                            formatted = ", ".join([repr(x) for x in rv2])
                                                        except Exception:
                                                            formatted = ", ".join([str(x) for x in rv2])
                                                        lines.append(f"{indent_str}    examples: [{formatted}]")
                                                        continue
                                                    if isinstance(rv2, (dict, list)):
                                                        lines.extend(
                                                            (
                                                                f"{indent_str}    {rk2}:",
                                                                CommandMetadataAPI._to_yaml_like(rv2, indent + 3, omit_command_name=omit_command_name),
                                                            )
                                                        )
                                                    else:
                                                        lines.append(f"{indent_str}    {rk2}: {rv2}")
                                            else:
                                                lines.append(f"{indent_str}  - {param}")
                                    elif isinstance(rv, dict):
                                        lines.extend(
                                            (
                                                f"{indent_str}  {('sample utterances' if rk == 'plain_utterances' else rk)}:",
                                                CommandMetadataAPI._to_yaml_like(rv, indent + 2, omit_command_name=omit_command_name),
                                            )
                                        )
                                    elif isinstance(rv, list):
                                        if not rv:
                                            continue
                                        lines.append(f"{indent_str}  {('sample utterances' if rk == 'plain_utterances' else rk)}:")
                                        for sub in rv:
                                            if isinstance(sub, dict):
                                                sub_keys = list(sub.keys())
                                                if not sub_keys:
                                                    continue
                                                fkey = sub_keys[0]
                                                fval = sub[fkey]
                                                if isinstance(fval, (dict, list)):
                                                    lines.extend(
                                                        (
                                                            f"{indent_str}  - {fkey}:",
                                                            CommandMetadataAPI._to_yaml_like(
                                                                fval, indent + 3, omit_command_name=omit_command_name
                                                            ),
                                                        )
                                                    )
                                                else:
                                                    lines.append(f"{indent_str}  - {fkey}: {fval}")
                                                for rkk in sub_keys[1:]:
                                                    rvv = sub[rkk]
                                                    if isinstance(rvv, (dict, list)):
                                                        lines.extend(
                                                            (
                                                                f"{indent_str}    {rkk}:",
                                                                CommandMetadataAPI._to_yaml_like(rvv, indent + 3, omit_command_name=omit_command_name),
                                                            )
                                                        )
                                                    else:
                                                        lines.append(f"{indent_str}    {rkk}: {rvv}")
                                            else:
                                                lines.append(f"{indent_str}  - {sub}")
                                    else:
                                        lines.append(f"{indent_str}  {('sample utterances' if rk == 'plain_utterances' else rk)}: {rv}")

                            elif not omit_command_name:
                                lines.append(f"{indent_str}- {item}")
                            # 6) Separator line between commands (but not after the last)
                            if idx != len(v) - 1:
                                lines.append(f"{indent_str}")
                    else:
                        if len(v) == 0:
                            continue
                        # lines.append(f"{indent_str}{display_key}:")
                        # Default list rendering for non-command lists
                        for item in v:
                            if isinstance(item, dict):
                                item_keys = list(item.keys())
                                if not item_keys:
                                    continue
                                first_key = item_keys[0]
                                first_val = item[first_key]
                                if isinstance(first_val, (dict, list)):
                                    lines.append(f"{indent_str}- {first_key}:")
                                    sub_item = {k2: item[k2] for k2 in item_keys if k2 != first_key}
                                    if first_val:
                                        lines.append(CommandMetadataAPI._to_yaml_like(first_val, indent + 2, omit_command_name=omit_command_name))
                                    if sub_item:
                                        lines.append(CommandMetadataAPI._to_yaml_like(sub_item, indent + 2, omit_command_name=omit_command_name))
                                else:
                                    lines.append(f"{indent_str}- {first_key}: {first_val}")
                                for rk in item_keys[1:]:
                                    rv = item[rk]
                                    if isinstance(rv, (dict, list)):
                                        lines.extend(
                                            (
                                                f"{indent_str}  {rk}:",
                                                CommandMetadataAPI._to_yaml_like(rv, indent + 2, omit_command_name=omit_command_name),
                                            )
                                        )
                                    else:
                                        lines.append(f"{indent_str}  {rk}: {rv}")
                            else:
                                lines.append(f"{indent_str}- {item}")
                else:
                    # lines.append(f"{indent_str}{display_key}: {v}")
                    lines.append(f"{indent_str}{v}")
            return "\n".join(lines)

        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    item_keys = list(item.keys())
                    if not item_keys:
                        continue
                    first_key = item_keys[0]
                    first_val = item[first_key]
                    if isinstance(first_val, (dict, list)):
                        lines.extend(
                            (
                                f"{indent_str}- {first_key}:",
                                CommandMetadataAPI._to_yaml_like(
                                    first_val, indent + 1, omit_command_name=omit_command_name
                                ),
                            )
                        )
                    else:
                        lines.append(f"{indent_str}- {first_key}: {first_val}")
                    for rk in item_keys[1:]:
                        rv = item[rk]
                        if isinstance(rv, (dict, list)):
                            lines.extend(
                                (
                                    f"{indent_str}  {rk}:",
                                    CommandMetadataAPI._to_yaml_like(rv, indent + 2, omit_command_name=omit_command_name),
                                )
                            )
                        else:
                            lines.append(f"{indent_str}  {rk}: {rv}")
                elif isinstance(item, list):
                    lines.extend(
                        (
                            f"{indent_str}-",
                            CommandMetadataAPI._to_yaml_like(item, indent + 1, omit_command_name=omit_command_name),
                        )
                    )
                else:
                    lines.append(f"{indent_str}- {item}")
            return "\n".join(lines)

        return f"{indent_str}{value}"

    @staticmethod
    def get_command_display_text(
        subject_workflow_path: str,
        cme_workflow_path: str,
        active_context_name: str,
        for_agents: bool = False,
    ) -> str:
        """
        Return a YAML-like display text for commands in the given context.

        - Calls get_enhanced_command_info to retrieve raw command metadata
        - Removes empty fields/lists and the 'default' field
        - Includes input 'pattern' only when for_agents=True
        """
        meta = CommandMetadataAPI.get_enhanced_command_info(
            subject_workflow_path=subject_workflow_path,
            cme_workflow_path=cme_workflow_path,
            active_context_name=active_context_name,
        )
        
        # Build minimal context info (inheritance/containment if available)
        context_info: Dict[str, Any] = {
            "name": active_context_name,
            "display_name": ("global" if active_context_name == "*" else active_context_name),
        }

        with contextlib.suppress(Exception):
            inheritance_path = Path(subject_workflow_path) / "context_inheritance_model.json"
            if inheritance_path.exists():
                with open(inheritance_path) as f:
                    inheritance_data = json.load(f)
                    if vals := inheritance_data.get(active_context_name, []):
                        context_info["inheritance"] = vals
            containment_path = Path(subject_workflow_path) / "context_containment_model.json"
            if containment_path.exists():
                with open(containment_path) as f:
                    containment_data = json.load(f)
                    if vals := containment_data.get(active_context_name, []):
                        context_info["containment"] = vals

        # Massage commands for display
        cmds = sorted(meta.get("commands", []), key=lambda x: x.get("name", ""))
        # If there are no commands, preserve original behavior of rendering an empty header
        if not cmds:
            # Preserve header even when no commands are available
            empty_display = {"context": context_info, "commands": []}
            empty_display = CommandMetadataAPI._prune_empty(empty_display, remove_keys={"default"})
            rendered = CommandMetadataAPI._to_yaml_like(empty_display, omit_command_name=False)
            # Ensure a visible header is present
            header = "Commands available:"
            if not rendered.strip():
                return header
            return f"{header}\n{rendered}"

        # Build the combined display by stitching per-command strings while keeping
        # a single "Commands available:" header and blank lines between commands
        parts: List[str] = []
        for cmd in cmds:
            fq = cmd.get("qualified_name", "")
            if part := CommandMetadataAPI.get_command_display_text_for_command(
                subject_workflow_path=subject_workflow_path,
                cme_workflow_path=cme_workflow_path,
                active_context_name=active_context_name,
                qualified_command_name=fq,
                for_agents=for_agents,
            ):
                parts.append(part)

        if not parts:
            empty_display = {"context": context_info, "commands": []}
            empty_display = CommandMetadataAPI._prune_empty(empty_display, remove_keys={"default"})
            return CommandMetadataAPI._to_yaml_like(empty_display, omit_command_name=False)

        combined_lines: List[str] = ["Commands available:"]
        for idx, text in enumerate(parts):
            lines = text.splitlines()
            if idx > 0:
                # Insert a blank line as separator (mirrors original formatter)
                combined_lines.append("")
            combined_lines.extend(lines)

        return "\n".join(combined_lines)

    @staticmethod
    def get_command_display_text_for_command(
        subject_workflow_path: str,
        cme_workflow_path: str,
        active_context_name: str,
        qualified_command_name: str,
        for_agents: bool = False,
        omit_command_name: bool = False
    ) -> str:
        """
        Return a YAML-like display text for a single command in the given context.

        Mirrors get_command_display_text but filters to a specific command only.
        """
        meta = CommandMetadataAPI.get_enhanced_command_info(
            subject_workflow_path=subject_workflow_path,
            cme_workflow_path=cme_workflow_path,
            active_context_name=active_context_name,
        )

        target_cmd: Dict[str, Any] | None = next(
            (
                cmd
                for cmd in meta.get("commands", [])
                if cmd.get("qualified_name") == qualified_command_name
            ),
            None,
        )
        if target_cmd is None:
            leaf = qualified_command_name.split("/")[-1]
            for cmd in meta.get("commands", []):
                if cmd.get("name") == leaf:
                    target_cmd = cmd
                    break

        # Build minimal context info (inheritance/containment if available)
        context_info: Dict[str, Any] = {
            "name": active_context_name,
            "display_name": ("global" if active_context_name == "*" else active_context_name),
        }
        with contextlib.suppress(Exception):
            inheritance_path = Path(subject_workflow_path) / "context_inheritance_model.json"
            if inheritance_path.exists():
                with open(inheritance_path) as f:
                    inheritance_data = json.load(f)
                    if vals := inheritance_data.get(active_context_name, []):
                        context_info["inheritance"] = vals
            containment_path = Path(subject_workflow_path) / "context_containment_model.json"
            if containment_path.exists():
                with open(containment_path) as f:
                    containment_data = json.load(f)
                    if vals := containment_data.get(active_context_name, []):
                        context_info["containment"] = vals

        # If command not found, return empty string (no header here)
        if target_cmd is None:
            return ""

        # Massage the single command for display (mirrors the formatter in the multi-command path)
        new_cmd: Dict[str, Any] = {}
        if "qualified_name" in target_cmd:
            new_cmd["qualified_name"] = target_cmd["qualified_name"]
        if "name" in target_cmd:
            new_cmd["name"] = target_cmd["name"]
        if doc_val := (target_cmd.get("doc_string") or "").strip():
            new_cmd["doc_string"] = doc_val

        inputs: List[Dict[str, Any]] = []
        for inp in target_cmd.get("inputs", []) or []:
            inp_out: Dict[str, Any] = {}
            if "name" in inp:
                inp_out["name"] = inp["name"]
            if "type" in inp:
                inp_out["type"] = CommandMetadataAPI._simplify_type_str(inp.get("type", ""))
            if desc := inp.get("description", ""):
                inp_out["description"] = desc
            if examples := inp.get("examples", []) or []:
                inp_out["examples"] = examples
            if (af := inp.get("available_from", None)) is not None:
                inp_out["available_from"] = af
            if for_agents:
                if pattern := inp.get("pattern", None):
                    if hasattr(pattern, 'pattern'):
                        pattern = pattern.pattern
                    inp_out["pattern"] = str(pattern)
            inputs.append(inp_out)
        if inputs:
            new_cmd["inputs"] = inputs

        outputs: List[Dict[str, Any]] = []
        for outp in target_cmd.get("outputs", []) or []:
            out_out: Dict[str, Any] = {}
            if "name" in outp:
                out_out["name"] = outp["name"]
            if "type" in outp:
                out_out["type"] = CommandMetadataAPI._simplify_type_str(outp.get("type", ""))
            if desc := outp.get("description", ""):
                out_out["description"] = desc
            if examples := outp.get("examples", []) or []:
                out_out["examples"] = examples
            if (af := outp.get("available_from", None)) is not None:
                out_out["available_from"] = af
            outputs.append(out_out)
        if outputs:
            new_cmd["outputs"] = outputs

        if utter := target_cmd.get("plain_utterances", []) or []:
            new_cmd["plain_utterances"] = utter[:2]

        display_obj: Dict[str, Any] = {
            "context": context_info,
            "commands": [new_cmd],
        }
        display_obj = CommandMetadataAPI._prune_empty(display_obj, remove_keys={"default"})
        return CommandMetadataAPI._to_yaml_like(display_obj, omit_command_name=omit_command_name)

    # ------------------------------------------------------------------
    # Bulk metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_params_for_all_commands(workflow_path: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Return a mapping of qualified command name -> {inputs: [...], outputs: [...]} where each param is a dict
        with name, type_str, description, examples.
        """
        directory = CommandDirectory.load(workflow_path)

        params_by_command: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for qualified_name in directory.get_commands():
            try:
                # Ensure metadata is hydrated and import module by file path
                directory.ensure_command_hydrated(qualified_name)
                cmd_meta = directory.get_command_metadata(qualified_name)
                module = python_utils.get_module(
                    cmd_meta.response_generation_module_path,
                    cmd_meta.workflow_folderpath or workflow_path,
                )
            except Exception:
                continue
            if not module or not hasattr(module, "Signature"):
                continue
            sig = module.Signature

            inputs: List[Dict[str, Any]] = []
            outputs: List[Dict[str, Any]] = []

            InputModel = getattr(sig, "Input", None)
            if InputModel is not None and hasattr(InputModel, "model_fields"):
                inputs.extend(
                    {
                        "name": name,
                        "type_str": str(getattr(field, "annotation", "")),
                        "description": str(
                            getattr(field, "description", "") or ""
                        ),
                        "examples": list(getattr(field, "examples", []) or []),
                        "default": getattr(field, "default", None),
                    }
                    for name, field in InputModel.model_fields.items()
                )
            OutputModel = getattr(sig, "Output", None)
            if OutputModel is not None and hasattr(OutputModel, "model_fields"):
                outputs.extend(
                    {
                        "name": name,
                        "type_str": str(getattr(field, "annotation", "")),
                        "description": str(
                            getattr(field, "description", "") or ""
                        ),
                        "examples": list(getattr(field, "examples", []) or []),
                    }
                    for name, field in OutputModel.model_fields.items()
                )
            if inputs or outputs:
                params_by_command[qualified_name] = {"inputs": inputs, "outputs": outputs}

        return params_by_command

    @staticmethod
    def get_all_commands_metadata(workflow_path: str) -> List[Dict[str, Any]]:
        """
        Return a list of command metadata dicts suitable for documentation generation:
          - command_name
          - file_path
          - plain_utterances
          - input_model (name if exists)
          - output_model (name if exists)
          - docstring (module docstring if present)
          - errors (empty list unless issues)
        """
        directory = CommandDirectory.load(workflow_path)

        metadata_list: List[Dict[str, Any]] = []
        for qualified_name in sorted(directory.get_commands()):
            meta = {
                "command_name": qualified_name.split("/")[-1],
                "file_path": None,
                "plain_utterances": [],
                "input_model": None,
                "output_model": None,
                "docstring": None,
                "errors": [],
            }
            try:
                cmd_meta = directory.get_command_metadata(qualified_name)
                meta["file_path"] = cmd_meta.response_generation_module_path

                if module := python_utils.get_module(
                    cmd_meta.response_generation_module_path,
                    cmd_meta.workflow_folderpath or workflow_path,
                ):
                    if module_doc := inspect.getdoc(module) or getattr(module, "__doc__", None):
                        meta["docstring"] = module_doc

                    sig = getattr(module, "Signature", None)
                    if sig is not None:
                        if hasattr(sig, "Input"):
                            meta["input_model"] = "Input"
                        if hasattr(sig, "Output"):
                            meta["output_model"] = "Output"
                        if hasattr(sig, "plain_utterances"):
                            meta["plain_utterances"] = list(getattr(sig, "plain_utterances") or [])

            except Exception as e:
                meta["errors"].append(str(e))

            metadata_list.append(meta)

        return metadata_list
