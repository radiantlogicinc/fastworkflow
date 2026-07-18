"""
Runtime insights distillation module for planning and execution agents.

Runs a teacher (large LLM) and student (small LLM) on the same user query,
compares their planning decisions and execution actions, and extracts insights
when the student diverges from the teacher.

Two types of insights are extracted:
- **Planning insights** ("what TO DO"): Prescriptive rules for the planner,
  stored in `planning_agent_insights.md`
- **Execution insights** ("what NOT to do"): Anti-patterns for the execution agent,
  stored in `execution_agent_anti_patterns.md`

Planning comparison uses the generated plans from `build_query_with_next_steps`.
Execution comparison uses actual resolved command_name and parameters from action.jsonl.
Full ReAct trajectories are passed to the insight extraction LLM for richer context.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import dspy

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_utils

# Per-turn log of executed actions, written by the workflow agent and read back
# here to compare teacher vs. student trajectories.
_ACTION_LOG_FILE = "action.jsonl"


def _announce(title: str, subtitle: str = "", style: str = "cyan") -> None:
    """
    Print an observability banner for a distillation phase.

    Distillation runs the agent twice (teacher then student) for a single user
    message, so without this the user cannot tell which model produced which
    output. Uses rich when available, else a plain print; never raises.
    """
    line = f"{title} — {subtitle}" if subtitle else title
    try:
        from rich.console import Console
        from rich.panel import Panel
        Console().print(Panel(line, style=style, expand=False))
    except Exception:
        print(f"\n=== {line} ===")


def _reset_action_log() -> None:
    """Remove the action log so the next agent pass starts from a clean slate."""
    if os.path.exists(_ACTION_LOG_FILE):
        os.remove(_ACTION_LOG_FILE)


@dataclass
class PlanningStep:
    """Captures a single planning decision during agent execution."""
    step_number: int           # For correlating multi-turn planning decisions
    user_query: str
    generated_plan: list[str]  # List of next steps from planner
    reasoning: str = ""        # Chain-of-thought reasoning from the planner


@dataclass
class DistillationResult:
    """Result of a distillation run for a single message."""
    command_output: fastworkflow.CommandOutput
    planning_insights_extracted: int = 0
    execution_insights_extracted: int = 0

    @property
    def insights_extracted(self) -> int:
        """Total insights (planning + execution) for backward compatibility."""
        return self.planning_insights_extracted + self.execution_insights_extracted


class PlanningInsightExtractionSignature(dspy.Signature):
    """You are analyzing why a student planner generated an inferior plan compared to the teacher.

    The planner's job is to break down a user request into a sequence of workflow commands.
    A good plan identifies the RIGHT commands in the RIGHT order to fulfill the user's intent.

    You are given:
    - The user's original query
    - Teacher's plan (the correct approach)
    - Student's plan (may have issues)
    - A summary of how the plans diverged
    - Teacher's executed actions (what the teacher actually did — the ground truth)
    - Student's executed actions (what the student actually did)

    Use the executed actions to validate whether a plan led to correct behavior.
    A plan that looks different but leads to the same correct actions may be acceptable.

    CRITICAL — Context divergence awareness:
    In multi-turn conversations, teacher and student may have different conversation
    history context from prior turns. This can legitimately cause different planning
    decisions. Before extracting a rule, analyze whether:
    1. The planning difference was a genuine mistake (extract a rule)
    2. The difference was justified by different context (return EMPTY)

    YOUR TASK:

    1. **Analyze and Find Root Cause**:
    - Analyze the context each planner had access to
    - Identify if there was a genuine planning mistake vs context-justified difference
    - Look at Step 0 (initial planning) - this is usually where things go wrong
    - What did the student plan that was incorrect?
    - What did the teacher plan that was correct?
    - What actions are MISSING from student's execution?

    2. **Generate 1-3 Specific, Actionable Rules if genuine mistakes are found**:
    - Rules should be PRESCRIPTIVE
    - Rules should reference SPECIFIC command names
    - Rules should prevent THIS EXACT failure from happening again
    - Focus on the ROOT CAUSE, not symptoms

    Return rules as: 1. [rule], 2. [rule], 3. [rule]
    Return EMPTY if:
    - Student's plan was reasonable given its context
    - Rules duplicate existing insights"""

    user_query: str = dspy.InputField()
    teacher_plan: str = dspy.InputField(
        desc="The correct plan generated by the teacher planner"
    )
    student_plan: str = dspy.InputField(
        desc="The plan generated by the student planner (may have issues)"
    )
    divergence_summary: str = dspy.InputField(
        desc="Summary of how teacher and student plans differ"
    )
    teacher_actions: str = dspy.InputField(
        desc="Actions actually executed by the teacher agent (command names + parameters)"
    )
    student_actions: str = dspy.InputField(
        desc="Actions actually executed by the student agent (command names + parameters)"
    )
    existing_insights: str = dspy.InputField(
        desc="Already known planning rules (avoid duplicates)"
    )
    insights: str = dspy.OutputField(
        desc="1-3 prescriptive, workflow-general rules as numbered list. "
             "Return EMPTY if no genuine mistakes, context-justified, or duplicates."
    )


class InsightExtractionSignature(dspy.Signature):
    """Analyze teacher vs student execution agent ReAct trajectories for the same user query.
    Both trajectories contain the full sequence of thoughts, tool calls, arguments, and
    observations (including any ask_user interactions and conversation history context).

    A divergence summary describes the concrete differences in executed actions
    (command names + parameters). Use the full trajectories to understand WHY
    each agent made its choices.

    CRITICAL — Multi-turn context awareness:
    In multi-turn conversations, teacher and student may have received different
    conversation context from prior turns (visible in observation_about_conversation_history
    and ask_user observations). This can legitimately cause different action sequences.
    Before extracting an insight, determine whether the student's actions were
    reasonable given the context it actually had. Only extract insights for
    genuine mistakes — NOT for context-justified differences.

    Return insights as single-line bullet points (one per line, prefixed with "- ").
    You may return multiple insights if there are multiple distinct mistakes.
    Return EMPTY if differences are not genuine mistakes or duplicate existing insights."""
    user_query: str = dspy.InputField()
    teacher_trajectory: str = dspy.InputField(
        desc="Teacher's full ReAct trajectory (reference/correct behavior)"
    )
    student_trajectory: str = dspy.InputField(
        desc="Student's full ReAct trajectory (may contain mistakes)"
    )
    divergence_summary: str = dspy.InputField(
        desc="Human-readable summary of action-level differences between teacher and student "
             "(which commands/params differed, ordering differences, etc.)"
    )
    existing_insights: str = dspy.InputField(
        desc="Already known anti-patterns (avoid duplicates)"
    )
    insights: str = dspy.OutputField(
        desc="One or more anti-pattern insights as bullet points (each line prefixed with '- '). "
             "Each insight should be a concise, actionable single-line point. "
             "Return EMPTY if no genuine mistakes found, differences are context-justified, "
             "or insights duplicate existing ones."
    )


class DistillationSession:
    """Orchestrates a single distillation comparison for one user message."""

    def __init__(self, wec: "fastworkflow.WorkflowExecutionContext"):
        # `wec` is the WorkflowExecutionContext (session engine). Distillation is
        # CLI/Topology-A only and drives its own agent passes over the WEC's state.
        self.chat_session = wec
        self.planning_insights_extracted: int = 0
        self.execution_insights_extracted: int = 0

    # ------------------------------------------------------------------
    # State snapshot / restore
    # ------------------------------------------------------------------

    def snapshot_workflow_state(self) -> dict:
        """Capture current workflow + CME state for later restoration."""
        workflow = self.chat_session.get_active_workflow()
        cme = self.chat_session.cme_workflow
        return {
            "workflow_dict": workflow._to_dict(),
            "cme_dict": cme._to_dict(),
            "conversation_history_len": len(
                self.chat_session.conversation_history.messages
            ),
        }

    def restore_workflow_state(self, snapshot: dict):
        """Restore workflow + CME state from a prior snapshot."""
        workflow = self.chat_session.get_active_workflow()
        wd = snapshot["workflow_dict"]
        workflow._context = wd["workflow_context"]
        workflow._is_complete = wd["is_complete"]
        workflow._save()
        workflow._dirty = False

        cme = self.chat_session.cme_workflow
        cd = snapshot["cme_dict"]
        cme._context = cd["workflow_context"]
        cme._save()
        cme._dirty = False

        orig_len = snapshot["conversation_history_len"]
        # Create a new History instance instead of modifying frozen messages
        import dspy
        self.chat_session._conversation_history = dspy.History(
            messages=self.chat_session._conversation_history.messages[:orig_len]
        )

        _reset_action_log()

    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_exec_steps(trajectory: dict) -> list[dict]:
        """
        Extract execute_workflow_query steps from a ReAct trajectory.

        Returns a list of dicts with keys: step_idx, thought, tool_name, tool_args, observation.
        Only includes steps where tool_name == "execute_workflow_query".
        """
        steps = []
        idx = 0
        while True:
            tool_name_key = f"tool_name_{idx}"
            if tool_name_key not in trajectory:
                break
            tool_name = trajectory[tool_name_key]
            if tool_name == "execute_workflow_query":
                steps.append({
                    "step_idx": idx,
                    "thought": trajectory.get(f"thought_{idx}", ""),
                    "tool_name": tool_name,
                    "tool_args": trajectory.get(f"tool_args_{idx}", {}),
                    "observation": trajectory.get(f"observation_{idx}", ""),
                })
            idx += 1
        return steps

    @staticmethod
    def _format_trajectory_for_llm(trajectory: dict) -> str:
        """Format a ReAct trajectory dict into a readable string for the insight LLM."""
        lines = []
        idx = 0
        while True:
            thought_key = f"thought_{idx}"
            if thought_key not in trajectory and f"tool_name_{idx}" not in trajectory:
                break
            if thought_key in trajectory:
                lines.append(f"[Step {idx}] Thought: {trajectory[thought_key]}")
            if f"tool_name_{idx}" in trajectory:
                lines.append(f"[Step {idx}] Tool: {trajectory[f'tool_name_{idx}']}")
            if f"tool_args_{idx}" in trajectory:
                args = trajectory[f"tool_args_{idx}"]
                lines.append(f"[Step {idx}] Args: {json.dumps(args, default=str)}")
            if f"observation_{idx}" in trajectory:
                obs = str(trajectory[f"observation_{idx}"])
                # Truncate very long observations
                if len(obs) > 500:
                    obs = f"{obs[:500]}... [truncated]"
                lines.append(f"[Step {idx}] Observation: {obs}")
            lines.append("")
            idx += 1

        # Include conversation history context if present
        if "observation_about_conversation_history" in trajectory:
            lines.insert(0, f"[Context] Conversation history: {trajectory['observation_about_conversation_history']}\n")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Trajectory comparison (uses action.jsonl records)
    # ------------------------------------------------------------------

    @staticmethod
    def _action_signature(action: dict) -> tuple[str, str]:
        """Return (command_name, sorted-params-json) as a comparable unit."""
        cmd = action.get("command_name", "")
        params = action.get("parameters", {})
        return (cmd, json.dumps(params, sort_keys=True, default=str))

    @staticmethod
    def _format_action(action: dict) -> str:
        """Human-readable representation of an action (command + params)."""
        cmd = action.get("command_name", "")
        params = action.get("parameters", {})
        return f"{cmd}({json.dumps(params, default=str)})" if params else cmd

    def compare_trajectories(
        self,
        teacher_actions: list[dict],
        student_actions: list[dict],
    ) -> tuple[bool, str]:
        """
        Compare teacher and student action lists from action.jsonl.

        Each action is treated as a (command_name, parameters) unit.
        Instead of strict step-by-step ordering, this produces a human-readable
        summary of all differences and lets the insight extraction LLM judge
        whether those differences constitute genuine mistakes or are justified
        by multi-turn conversation context divergence.

        Returns:
            (has_divergence: bool, divergence_summary: str)
            divergence_summary is empty when has_divergence is False.
        """
        # Filter out internal error correction actions (abort commands from loop detection)
        def is_valid_action(a: dict) -> bool:
            cmd = a.get("command_name", "")
            # Exclude ErrorCorrection/abort and ask_user records (which have agent_query key)
            return cmd and not cmd.startswith("ErrorCorrection/") and "agent_query" not in a

        teacher_actions = [a for a in teacher_actions if is_valid_action(a)]
        student_actions = [a for a in student_actions if is_valid_action(a)]

        teacher_sigs = [self._action_signature(a) for a in teacher_actions]
        student_sigs = [self._action_signature(a) for a in student_actions]

        # Fast path: identical sequences (same actions, same order)
        if teacher_sigs == student_sigs:
            return False, ""

        differences: list[str] = []

        # Compare actions as (command_name, params) units
        teacher_sig_set = set(teacher_sigs)
        student_sig_set = set(student_sigs)

        only_teacher = teacher_sig_set - student_sig_set
        only_student = student_sig_set - teacher_sig_set

        if only_teacher:
            only_teacher_strs = [
                self._format_action(a)
                for a in teacher_actions
                if self._action_signature(a) in only_teacher
            ]
            differences.append(
                f"Actions executed only by teacher: {only_teacher_strs}"
            )

        if only_student:
            only_student_strs = [
                self._format_action(a)
                for a in student_actions
                if self._action_signature(a) in only_student
            ]
            differences.append(
                f"Actions executed only by student: {only_student_strs}"
            )

        # Check ordering of shared actions (only if there are actual shared actions)
        shared_teacher_sigs = [s for s in teacher_sigs if s in student_sig_set]
        shared_student_sigs = [s for s in student_sigs if s in teacher_sig_set]

        # Only report ordering differences if there are meaningful shared actions
        if shared_teacher_sigs and shared_student_sigs and shared_teacher_sigs != shared_student_sigs:
            teacher_order = [
                self._format_action(a)
                for a in teacher_actions
                if self._action_signature(a) in student_sig_set
            ]
            student_order = [
                self._format_action(a)
                for a in student_actions
                if self._action_signature(a) in teacher_sig_set
            ]
            # Only add if we actually have formatted actions (not empty strings)
            if teacher_order and student_order and any(teacher_order) and any(student_order):
                differences.append(
                    f"Different execution order — "
                    f"teacher: {teacher_order}, student: {student_order}"
                )

        if not differences:
            # Sequences differ in some way we didn't classify—still flag it
            differences.append(
                "Action sequences differ (unclassified difference)"
            )

        return True, "\n".join(differences)

    # ------------------------------------------------------------------
    # Planning trace helpers
    # ------------------------------------------------------------------

    def compare_planning_traces(
        self,
        teacher_steps: list[PlanningStep],
        student_steps: list[PlanningStep],
    ) -> tuple[bool, str]:
        """
        Compare teacher and student planning traces.

        Returns:
            (has_divergence: bool, divergence_summary: str)
        """
        if not teacher_steps and not student_steps:
            return False, ""

        differences = []

        # Compare plans at each step
        for i in range(max(len(teacher_steps), len(student_steps))):
            t_plan = teacher_steps[i].generated_plan if i < len(teacher_steps) else []
            s_plan = student_steps[i].generated_plan if i < len(student_steps) else []

            # Normalize for comparison
            t_normalized = [p.lower().strip() for p in t_plan]
            s_normalized = [p.lower().strip() for p in s_plan]

            if t_normalized != s_normalized:
                differences.append(
                    f"Step {i}: Teacher planned {t_plan}, Student planned {s_plan}"
                )

        return (True, "\n".join(differences)) if differences else (False, "")

    @staticmethod
    def _format_planning_traces_for_llm(steps: list[PlanningStep]) -> str:
        """Format planning steps into readable string for insight LLM."""
        lines = []
        for step in steps:
            lines.append(f"[Step {step.step_number}] Query: {step.user_query}")
            if step.reasoning:
                lines.append(f"[Step {step.step_number}] Reasoning: {step.reasoning}")
            lines.extend((f"[Step {step.step_number}] Plan: {step.generated_plan}", ""))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Agent run helper
    # ------------------------------------------------------------------

    def _run_agent_pass(
        self,
        message: str,
        agent_lm_role: str,
        agent_api_key_role: str,
        planner_lm_role: str,
        planner_api_key_role: str,
    ) -> tuple[fastworkflow.CommandOutput, dict, list[dict], list[PlanningStep]]:
        """
        Run a full agent pass with the specified LLMs for planner and agent.

        Returns:
            (command_output, trajectory_dict, actions_from_jsonl, planning_steps)
            actions_from_jsonl: list of dicts from action.jsonl with keys
                command, command_name, parameters, response
            planning_steps: list of PlanningStep objects capturing planning decisions
        """
        # Clean prior action log
        _reset_action_log()

        # Store raw message in workflow context (mirrors _process_agent_message)
        self.chat_session.get_active_workflow().context["raw_user_message"] = message

        # Create agent with the specified LLM
        from fastworkflow.workflow_agent import (
            initialize_workflow_tool_agent,
            build_query_with_next_steps,
            _what_can_i_do,
        )

        # Load execution insights for the agent
        execution_insights = getattr(
            self.chat_session, "_execution_insights", None
        )

        agent = initialize_workflow_tool_agent(
            self.chat_session,
            execution_insights=execution_insights,
        )

        # Temporarily install this agent
        original_agent = self.chat_session._workflow_tool_agent
        self.chat_session._workflow_tool_agent = agent

        try:
            refined_message = self.chat_session._refine_user_query(
                message, self.chat_session.conversation_history
            )

            # Get planning insights for injection into planner prompt
            planning_insights = getattr(self.chat_session, '_planning_insights', None)

            # Set up planner LM for this pass
            planner_lm = dspy_utils.get_lm(planner_lm_role, planner_api_key_role)

            # Store planner_lm on the session so it can be used for replanning
            self.chat_session._current_planner_lm = planner_lm

            # Initialize capture list on the session to capture ALL plans
            # (initial + replanning during agent execution)
            self.chat_session._planning_steps_capture = []

            # Build initial query with next steps using the PLANNER LLM
            # The hook in build_query_with_next_steps will auto-capture the plan
            command_info = build_query_with_next_steps(
                refined_message, self.chat_session, planning_insights=planning_insights, planner_lm=planner_lm
            )

            # Get available commands for current context
            available_commands = _what_can_i_do(self.chat_session)

            # Run the agent with the specified AGENT LLM, reusing the WEC's shared
            # agent-invocation contract (dspy.context + AdapterParseError retry).
            agent_lm = dspy_utils.get_lm(agent_lm_role, agent_api_key_role)
            agent_result = self.chat_session._call_agent_with_retry(
                lambda: agent(
                    user_query=command_info,
                    available_commands=available_commands,
                ),
                lm=agent_lm,
            )

            # Extract result text
            result_text = (
                agent_result.final_answer
                if hasattr(agent_result, "final_answer")
                else str(agent_result)
            )

            # Build CommandOutput
            command_response = fastworkflow.CommandResponse(response=result_text)
            command_output = fastworkflow.CommandOutput(
                command_responses=[command_response]
            )
            command_output.workflow_name = (
                self.chat_session.get_active_workflow().folderpath.split("/")[-1]
            )

            # Read actions from the action log (actual resolved command_name + parameters)
            actions = []
            if os.path.exists(_ACTION_LOG_FILE):
                with open(_ACTION_LOG_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        if line := line.strip():
                            try:
                                actions.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

            # Capture the full ReAct trajectory
            trajectory = dict(agent.current_trajectory)

            self.chat_session.summarize_and_record_turn(message, actions, result_text)

            # Flush workflow state
            if workflow := self.chat_session.get_active_workflow():
                workflow.flush()

            # Collect all captured planning steps (initial + replanning)
            planning_steps = list(
                getattr(self.chat_session, '_planning_steps_capture', [])
            )

            return command_output, trajectory, actions, planning_steps

        finally:
            # Restore original agent and clean up distillation-specific attributes
            self.chat_session._workflow_tool_agent = original_agent
            if hasattr(self.chat_session, '_current_planner_lm'):
                delattr(self.chat_session, '_current_planner_lm')
            if hasattr(self.chat_session, '_planning_steps_capture'):
                delattr(self.chat_session, '_planning_steps_capture')

    # ------------------------------------------------------------------
    # Insight extraction
    # ------------------------------------------------------------------

    def extract_insights(
        self,
        teacher_traj: dict,
        student_traj: dict,
        divergence_summary: str,
        user_query: str,
    ) -> list[str]:
        """
        Use LLM_DISTILLATION to analyze the trajectory delta and extract insights.

        Uses the full ReAct trajectories (which include conversation context via
        observation_about_conversation_history and ask_user observations).

        The divergence_summary describes concrete action-level differences;
        the LLM uses the full trajectories to judge whether differences are
        genuine mistakes or context-justified.

        Returns a list of insight strings, or empty list if no genuine mistakes.
        """
        # Load existing insights for dedup
        from fastworkflow.utils.insights_loader import load_workflow_insights

        workflow = self.chat_session.get_active_workflow()
        existing_insights = (
            load_workflow_insights(workflow.folderpath, "execution_agent") or ""
        )

        # Format trajectories for the LLM
        teacher_formatted = self._format_trajectory_for_llm(teacher_traj)
        student_formatted = self._format_trajectory_for_llm(student_traj)

        lm = dspy_utils.get_lm("LLM_DISTILLATION", "LITELLM_API_KEY_DISTILLATION")

        with dspy.context(lm=lm):
            extractor = dspy.ChainOfThought(InsightExtractionSignature)
            result = extractor(
                user_query=user_query,
                teacher_trajectory=teacher_formatted,
                student_trajectory=student_formatted,
                divergence_summary=divergence_summary,
                existing_insights=existing_insights,
            )

        raw_insights = (getattr(result, "insights", None) or "").strip()

        # Treat "EMPTY" or empty string as no insights
        if not raw_insights or raw_insights.upper() == "EMPTY":
            return []

        # Parse bullet points: each line starting with "- "
        insights = []
        for line in raw_insights.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                insights.append(line[2:].strip())
            elif line and not line.startswith("#"):
                # Accept non-prefixed lines as insights too
                insights.append(line)

        return [i for i in insights if i]

    # ------------------------------------------------------------------
    # Insight persistence
    # ------------------------------------------------------------------

    def _insights_dir(self) -> Path:
        """Return (creating if needed) the workflow's Insights directory."""
        workflow = self.chat_session.get_active_workflow()
        workflow_name = Path(workflow.folderpath).name
        insights_dir = Path(workflow.folderpath) / "Insights" / workflow_name
        insights_dir.mkdir(parents=True, exist_ok=True)
        return insights_dir

    @staticmethod
    def _append_numbered_insights(
        insights: list[str],
        insights_file: Path,
        header: str,
        number_pattern: str,
        entry_format: str,
    ) -> None:
        """
        Append ``insights`` to ``insights_file`` as a numbered list, continuing
        the numbering already present in the file.

        ``number_pattern`` is a regex (with one capturing group for the number)
        used to find existing entries; ``entry_format`` is a format string taking
        ``num`` and ``insight``. The two MUST correspond so the numbering the file
        is read with matches the numbering it is written with.
        """
        if insights_file.exists():
            content = insights_file.read_text(encoding="utf-8")
            numbers = re.findall(number_pattern, content, re.MULTILINE)
            next_num = max(int(n) for n in numbers) + 1 if numbers else 1
        else:
            insights_file.write_text(header, encoding="utf-8")
            next_num = 1

        with open(insights_file, "a", encoding="utf-8") as f:
            for insight in insights:
                f.write(entry_format.format(num=next_num, insight=insight))
                next_num += 1

    def append_insights(self, insights: list[str]):
        """Append new insights to execution_agent_anti_patterns.md."""
        if not insights:
            return

        insights_file = self._insights_dir() / "execution_agent_anti_patterns.md"
        self._append_numbered_insights(
            insights,
            insights_file,
            header=(
                "# Execution Agent Anti-Patterns\n\n"
                "Critical mistakes to avoid when executing workflows, "
                "derived from distillation.\n\n"
            ),
            number_pattern=r"^(\d+)\.\s",
            entry_format="{num}. {insight}\n",
        )

        self.execution_insights_extracted += len(insights)

    # ------------------------------------------------------------------
    # Planning insight extraction
    # ------------------------------------------------------------------

    def extract_planning_insights(
        self,
        teacher_steps: list[PlanningStep],
        student_steps: list[PlanningStep],
        divergence_summary: str,
        user_query: str,
        teacher_actions: list[dict],
        student_actions: list[dict],
    ) -> list[str]:
        """Extract prescriptive planning rules using LLM_DISTILLATION.

        Uses both planning traces and executed actions to give the insight
        extraction LLM full context for judging plan quality.
        """
        # Load existing planning insights for dedup
        from fastworkflow.utils.insights_loader import load_workflow_insights

        workflow = self.chat_session.get_active_workflow()
        existing_insights = (
            load_workflow_insights(workflow.folderpath, "planning_agent") or ""
        )

        # Format planning traces
        teacher_plan_str = self._format_planning_traces_for_llm(teacher_steps)
        student_plan_str = self._format_planning_traces_for_llm(student_steps)

        # Format executed actions for context
        teacher_actions_str = "\n".join(
            self._format_action(a) for a in teacher_actions
        ) or "(no actions executed)"
        student_actions_str = "\n".join(
            self._format_action(a) for a in student_actions
        ) or "(no actions executed)"

        lm = dspy_utils.get_lm("LLM_DISTILLATION", "LITELLM_API_KEY_DISTILLATION")

        with dspy.context(lm=lm):
            extractor = dspy.ChainOfThought(PlanningInsightExtractionSignature)
            result = extractor(
                user_query=user_query,
                teacher_plan=teacher_plan_str,
                student_plan=student_plan_str,
                divergence_summary=divergence_summary,
                teacher_actions=teacher_actions_str,
                student_actions=student_actions_str,
                existing_insights=existing_insights,
            )

        raw_insights = (getattr(result, "insights", None) or "").strip()
        if not raw_insights or raw_insights.upper() == "EMPTY":
            return []

        # Parse numbered rules (1. rule, 2. rule, etc.)
        insights = []
        for line in raw_insights.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line[:3]:
                if rule_text := line.split(".", 1)[1].strip():
                    insights.append(rule_text)

        return insights

    def append_planning_insights(self, insights: list[str]):
        """Append new planning insights to planning_agent_insights.md."""
        if not insights:
            return

        insights_file = self._insights_dir() / "planning_agent_insights.md"
        self._append_numbered_insights(
            insights,
            insights_file,
            header=(
                "# Planning Agent Insights\n\n"
                "Key insights for planning workflow execution strategies, "
                "derived from distillation training.\n\n"
            ),
            number_pattern=r"^## (\d+)\.",
            entry_format="## {num}. {insight}\n\n",
        )

        self.planning_insights_extracted += len(insights)


# ------------------------------------------------------------------
# Top-level orchestrator
# ------------------------------------------------------------------


def distill_message(
    chat_session: "fastworkflow.WorkflowExecutionContext", message: str
) -> DistillationResult:
    """
    Run teacher and student agents, extract BOTH planning and execution insights.

    Flow:
    1. Snapshot state
    2. Run teacher - capture planning traces + actions
    3. Save teacher's final state
    4. Restore pre-teacher state
    5. Run student - capture planning traces + actions
    6a. Compare planning traces → extract planning insights
    6b. Compare executed actions → extract execution insights
    7. On divergence: restore teacher's state; else keep student's
    8. Return teacher's output
    """
    ds = DistillationSession(chat_session)

    teacher_model = fastworkflow.get_env_var("LLM_TEACHER_AGENT") or "LLM_TEACHER_AGENT"
    student_model = fastworkflow.get_env_var("LLM_STUDENT_AGENT") or "LLM_STUDENT_AGENT"

    # 1. Snapshot initial state
    initial_snapshot = ds.snapshot_workflow_state()

    # 2. Run teacher (returns planning_steps too now)
    _announce("TEACHER pass", teacher_model, style="bold magenta")
    teacher_output, teacher_traj, teacher_actions, teacher_plans = ds._run_agent_pass(
        message,
        agent_lm_role="LLM_TEACHER_AGENT",
        agent_api_key_role="LITELLM_API_KEY_TEACHER_AGENT",
        planner_lm_role="LLM_TEACHER_PLANNER",
        planner_api_key_role="LITELLM_API_KEY_TEACHER_PLANNER",
    )

    # 3. Save teacher's final state
    teacher_final_state = ds.snapshot_workflow_state()

    # 4. Restore initial state for student run
    ds.restore_workflow_state(initial_snapshot)

    # 5. Run student
    _announce("STUDENT pass", student_model, style="bold cyan")
    try:
        student_output, student_traj, student_actions, student_plans = ds._run_agent_pass(
            message,
            agent_lm_role="LLM_STUDENT_AGENT",
            agent_api_key_role="LITELLM_API_KEY_STUDENT_AGENT",
            planner_lm_role="LLM_STUDENT_PLANNER",
            planner_api_key_role="LITELLM_API_KEY_STUDENT_PLANNER",
        )
    except Exception as e:
        logger.warning(f"Distillation: student agent failed: {e}")
        ds.restore_workflow_state(teacher_final_state)
        return DistillationResult(command_output=teacher_output)

    any_divergence = False

    # 6a. Compare and extract PLANNING insights
    planning_diverged, planning_summary = ds.compare_planning_traces(
        teacher_plans, student_plans
    )
    if planning_diverged:
        planning_insights = ds.extract_planning_insights(
            teacher_plans, student_plans,
            planning_summary, message,
            teacher_actions, student_actions,
        )
        if planning_insights:
            ds.append_planning_insights(planning_insights)
        any_divergence = True

    # 6b. Compare and extract EXECUTION insights (existing logic)
    exec_diverged, exec_summary = ds.compare_trajectories(
        teacher_actions, student_actions
    )
    if exec_diverged:
        exec_insights = ds.extract_insights(
            teacher_traj, student_traj,
            exec_summary, message
        )
        if exec_insights:
            ds.append_insights(exec_insights)
        any_divergence = True

    # 7. State restoration
    if any_divergence:
        ds.restore_workflow_state(teacher_final_state)
    # else: keep student's state (equivalent to teacher's)

    if any_divergence:
        _announce(
            "DISTILLATION: divergence found",
            f"{ds.planning_insights_extracted} planning + "
            f"{ds.execution_insights_extracted} execution insight(s) extracted",
            style="bold yellow",
        )
    else:
        _announce(
            "DISTILLATION: no divergence",
            "student matched teacher — no insights extracted",
            style="green",
        )

    return DistillationResult(
        command_output=teacher_output,
        planning_insights_extracted=ds.planning_insights_extracted,
        execution_insights_extracted=ds.execution_insights_extracted,
    )
