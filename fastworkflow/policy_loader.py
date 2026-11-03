"""Load a workflow's policy table from the settings.

Two settings (env file or process environment), both defaulting to a no-op,
because a policy that turns itself on is not a feature flag:

  FW_POLICY_MODE   off | enforce                       (default: off)
  FW_POLICY_TABLE  path/to/table.py[:ATTR]  or  package.module[:ATTR]
                   (ATTR defaults to TABLE; a relative path is tried against the
                   current directory, then against the workflow folder)

The table module exposes ``TABLE`` (a ``DecisionTable``) and optionally ``FACTS``
(``ContractFacts``). ``off`` is byte-for-byte the unmodified behaviour.

A table that cannot be loaded while the mode is not ``off`` is a HARD ERROR,
deliberately: a run configured to enforce that quietly enforced nothing would be
recorded, and measured, as if it had.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
from pathlib import Path
from typing import Optional

import fastworkflow
from fastworkflow.policy_decision import (
    NO_OP_TABLE,
    ContractFacts,
    DecisionTable,
    PolicyDecisionPoint,
    PolicyMode,
)
from fastworkflow.utils.logging import logger

MODE_SETTING = "FW_POLICY_MODE"
TABLE_SETTING = "FW_POLICY_TABLE"


def policy_mode() -> PolicyMode:
    """The configured mode; raises ValueError for a value that is not one."""
    raw = (fastworkflow.get_env_var(MODE_SETTING, default="") or "off").strip().lower()
    try:
        return PolicyMode(raw)
    except ValueError:
        raise ValueError(
            f"{MODE_SETTING}={raw!r} is not one of {[m.value for m in PolicyMode]}"
        ) from None


def load_table_module(reference: str, workflow_folderpath: Optional[str] = None):
    """(module, attribute name) for a ``path.py[:ATTR]`` or ``module[:ATTR]``."""
    module_ref, _, attribute = reference.strip().partition(":")
    attribute = attribute or "TABLE"
    if module_ref.endswith(".py"):
        path = Path(module_ref)
        if not path.is_absolute() and not path.is_file() and workflow_folderpath:
            path = Path(workflow_folderpath) / module_ref
        if not path.is_file():
            raise ValueError(
                f"{TABLE_SETTING} points at {module_ref}, which does not exist"
                + (f" (also tried under {workflow_folderpath})" if workflow_folderpath else "")
            )
        spec = importlib.util.spec_from_file_location(f"fw_policy_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    return module, attribute


def load_policy(
    workflow_folderpath: Optional[str] = None,
    *,
    mode: Optional[PolicyMode] = None,
    reference: Optional[str] = None,
) -> tuple[PolicyDecisionPoint, ContractFacts]:
    """The decision point and the workflow's facts, from the settings.

    ``mode`` / ``reference`` override the settings (tests, the replay audit).
    """
    mode = policy_mode() if mode is None else mode
    if mode is PolicyMode.OFF:
        return PolicyDecisionPoint(NO_OP_TABLE, PolicyMode.OFF), ContractFacts()
    reference = (
        reference
        if reference is not None
        else (fastworkflow.get_env_var(TABLE_SETTING, default="") or "")
    ).strip()
    if not reference:
        raise ValueError(
            f"{MODE_SETTING}={mode.value} with no {TABLE_SETTING}. The mechanism is "
            "framework-side but the rules are workflow content, so there is "
            "nothing to enforce."
        )
    module, attribute = load_table_module(reference, workflow_folderpath)
    table = getattr(module, attribute, None)
    if not isinstance(table, DecisionTable):
        raise ValueError(
            f"{TABLE_SETTING}={reference}: {attribute} is "
            f"{type(table).__name__}, not a DecisionTable"
        )
    facts = getattr(module, "FACTS", None) or ContractFacts()
    if not isinstance(facts, ContractFacts):
        raise ValueError(f"{TABLE_SETTING}={reference}: FACTS is not ContractFacts")
    facts = with_workflow_identifiers(facts, workflow_folderpath)
    logger.info(
        f"Policy {mode.value}: table {table.version} ({len(table.rows)} rows) "
        f"from {reference}"
    )
    return PolicyDecisionPoint(table, mode), facts


def with_workflow_identifiers(
    facts: ContractFacts, workflow_folderpath: Optional[str]
) -> ContractFacts:
    """``facts`` with ``id_patterns`` taken from the workflow when it gives none.

    A workflow may already declare its identifier formats for the split merge
    (``identifier_patterns`` in ``_skills/_split.py``); a table need not repeat them.
    """
    if facts.id_patterns or not workflow_folderpath:
        return facts
    try:
        from fastworkflow.workflow_agent import identifier_patterns

        patterns = tuple(p.pattern for p in identifier_patterns(workflow_folderpath))
    except Exception as exc:  # noqa: BLE001 - the split file reports its own errors
        logger.warning(f"Policy: workflow identifier patterns unavailable: {exc}")
        return facts
    return dataclasses.replace(facts, id_patterns=patterns) if patterns else facts


def attach_policy(agent, host) -> None:
    """Give a ReAct agent the configured policy. A no-op when the mode is off.

    ``host`` is the WorkflowExecutionContext the agent serves; the table is
    resolved against its app workflow's folder.
    """
    from fastworkflow.policy_runtime import AgentPolicy

    workflow = getattr(host, "_app_workflow", None)
    point, facts = load_policy(getattr(workflow, "folderpath", None))
    if point.enabled:
        agent.policy = AgentPolicy(point, facts, host=host)
