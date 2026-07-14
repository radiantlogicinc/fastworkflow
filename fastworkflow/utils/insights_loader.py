"""
Utility module for loading workflow-specific insights for knowledge distillation.

This module provides functionality to load planning and execution insights from
workflow directories, enabling the integration of learned patterns and anti-patterns
into agent prompts.
"""

from pathlib import Path
from typing import Optional
from fastworkflow.utils.logging import logger


def load_workflow_insights(workflow_folderpath: str, insight_type: str) -> Optional[str]:
    """
    Load workflow-specific insights for knowledge distillation.

    This function looks for insights files in the workflow's Insights directory,
    following the convention: workflow_folderpath/Insights/<workflow_name>/<insight_file>

    Args:
        workflow_folderpath: Absolute path to workflow directory
        insight_type: Type of insights to load. Supported values:
            - 'planning_agent': Planning strategies and patterns
            - 'execution_agent': Execution anti-patterns and rules

    Returns:
        Insights content as string if found, None otherwise

    Example:
        >>> insights = load_workflow_insights('/path/to/my_workflow', 'planning_agent')
        >>> if insights:
        ...     print(f"Loaded {len(insights)} characters of insights")
    """
    # Extract workflow name from folderpath (last component)
    workflow_name = Path(workflow_folderpath).name

    # Build path: workflow_folderpath/Insights/<workflow_name>/
    insights_dir = Path(workflow_folderpath) / "Insights" / workflow_name

    if not insights_dir.exists():
        logger.debug(f"No Insights directory found at {insights_dir}")
        return None

    # Map insight type to filename
    filename_map = {
        "planning_agent": "planning_agent_insights.md",
        "execution_agent": "execution_agent_anti_patterns.md"
    }

    if insight_type not in filename_map:
        logger.warning(f"Unknown insight type: {insight_type}. Supported types: {list(filename_map.keys())}")
        return None

    insight_file = insights_dir / filename_map[insight_type]

    if not insight_file.exists():
        logger.debug(f"Insights file not found: {insight_file}")
        return None

    try:
        content = insight_file.read_text(encoding='utf-8')
        logger.info(f"Loaded {insight_type} insights from {insight_file} ({len(content)} characters)")
        return content
    except Exception as e:
        logger.error(f"Error reading insights file {insight_file}: {e}")
        return None
