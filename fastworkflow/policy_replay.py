"""Replay recorded runs through a policy table, offline.

For every attempt of an experiment, walks the recorded steps in order, rebuilds
what the policy knew at that moment (commands run and what they returned, what
the user said), and asks the table what it would decide about the step the agent
actually took. Reports every step where the table would intervene, next to what
the recording shows.

This is how a table is audited before it is turned on: on passing trials it
should stay silent; on failing trials it should fire where the trial went
wrong, and nowhere else.

The replay drives the same ``AgentPolicy`` the live loop uses, so the valve, the
finish bounds and the recording rules are the live ones. Each attempt is
replayed on its own (one conversation, one history): merging a task's trials
would let one trial's reads ground another's writes.

Without ``live``, nothing that needs a model call is replayed: question
judgements are absent (rows that need one stay silent), consent is taken as
given, and intent is the one recorded on the run's own policy steps, if any.
With ``live``, the policy model reads every recorded user message and judges
every recorded question, exactly as the loop would (one call each).

A step that was blocked in the recording never ran, so it is not recorded as
run. If the new table would let it through, the replay says so (LIFTED) and
carries on without knowing its result.

    fastworkflow policy replay <workflow> [env] [passwords] \\
        --experiment EXP --table path/to/table.py [--task T ...] [--live]
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

from fastworkflow import distillation_trace as dtrace
from fastworkflow import tracing
from fastworkflow.policy_decision import (
    ASK_USER,
    ContractFacts,
    DecisionTable,
    PolicyDecisionPoint,
    PolicyMode,
)
from fastworkflow.policy_intent import parse_described
from fastworkflow.policy_runtime import AgentPolicy

_COMMAND_TOOL = "execute_workflow_query"


@dataclass(frozen=True)
class ReplayedStep:
    """One step where the recording or the replayed table intervened."""

    task_id: str
    attempt: int
    step_index: Any
    tool: str
    label: str
    recorded: Optional[str]
    replayed: Optional[str]
    rewrite: str = ""

    @property
    def mark(self) -> str:
        """same / LIFTED (recorded block the table no longer makes) / ADDED / CHANGED."""
        if self.recorded == self.replayed:
            return "same"
        if self.recorded and not self.replayed:
            return "LIFTED"
        if self.replayed and not self.recorded:
            return "ADDED"
        return "CHANGED"


@dataclass
class AttemptReplay:
    task_id: str
    attempt: int
    passed: Optional[bool]
    steps: list[ReplayedStep] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def fires(self) -> int:
        return sum(1 for s in self.steps if s.replayed)


@dataclass
class ReplayReport:
    experiment_id: str
    table_version: str
    live: bool
    attempts: list[AttemptReplay] = field(default_factory=list)

    def fires_on(self, passed: Optional[bool]) -> int:
        """Replayed blocks on attempts that passed (True), failed (False), or ungraded."""
        return sum(a.fires for a in self.attempts if a.passed is passed)


def _children(spans: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for span in spans:
        if span.get("parent_span_id"):
            out.setdefault(span["parent_span_id"], []).append(span)
    return out


def _attrs(span: dict) -> dict:
    return dtrace._attributes(span)


def _value(span: dict, key: str, default: Any = None) -> Any:
    value = _attrs(span).get(key, default)
    return dtrace.unwrap(value) if value is not None else default


def _executed(
    step: dict, children: dict[str, list[dict]]
) -> Optional[tuple[str, dict, bool, str]]:
    """(command name, parameters, success, response) of the command a step dispatched."""
    for call in children.get(step.get("span_id"), ()):
        if call.get("name") != tracing.SPAN_AGENT_TOOL_CALL:
            continue
        success = _attrs(call).get("success")
        for run in children.get(call.get("span_id"), ()):
            if run.get("name") == tracing.SPAN_COMMAND_EXECUTE and run.get("command_name"):
                params = _attrs(run).get("parameters")
                response = _value(run, "response_text") or _value(call, "response_text") or ""
                return (
                    run["command_name"],
                    params if isinstance(params, dict) else {},
                    bool(success) if success is not None else True,
                    str(response),
                )
    return None


def replay_attempt(
    store: Any,
    experiment_id: str,
    task_id: str,
    attempt: int,
    table: DecisionTable,
    facts: ContractFacts,
    *,
    live: bool = False,
    passed: Optional[bool] = None,
) -> AttemptReplay:
    """Replay one recorded attempt through ``table``."""
    result = AttemptReplay(task_id=task_id, attempt=attempt, passed=passed)
    rows = store.list_turns(
        experiment_id=experiment_id, task_id=task_id, attempt=attempt, limit=10000
    )
    if not rows:
        result.error = "no turns recorded"
        return result
    rows.sort(key=lambda r: (r.get("ordinal") or 0, r.get("turn_key") or ""))
    spans_by_turn = store.spans_for_turns([r["turn_key"] for r in rows])

    policy = AgentPolicy(
        PolicyDecisionPoint(table, PolicyMode.ENFORCE),
        facts if live else dataclasses.replace(facts, judge_questions=False),
        reads_with_model=live,
    )
    for row in rows:
        spans = sorted(spans_by_turn.get(row["turn_key"], []),
                       key=lambda s: s.get("start_ns") or 0)
        children = _children(spans)
        message = next(
            (_value(s, "user_message") for s in spans if s.get("name") == tracing.SPAN_TURN),
            None,
        ) or row.get("user_message") or ""
        policy.observe_turn(str(message))
        answer = str(row.get("answer") or "")
        pending_question = ""
        for span in spans:
            name = span.get("name")
            if name == tracing.SPAN_ASK_USER:
                reply = _value(span, "user_response")
                if reply is not None:
                    question = _value(span, "agent_query") or pending_question
                    policy.observe_reply(str(question or ""), str(reply))
                continue
            if name != tracing.SPAN_AGENT_STEP:
                continue
            attrs = _attrs(span)
            tool = attrs.get("tool_name")
            if not tool:
                continue
            if not live and attrs.get("policy_intent"):
                policy.state.intent = list(parse_described(attrs["policy_intent"]))
            recorded = attrs.get("policy_source")
            args = attrs.get("tool_args") if isinstance(attrs.get("tool_args"), dict) else {}
            decision = None
            executed = None
            if tool == _COMMAND_TOOL:
                executed = _executed(span, children)
                command = str(args.get("command", ""))
                if executed is not None or recorded:
                    name_, params = (executed[0], executed[1]) if executed else (
                        (command.split() or [""])[0], {})
                    policy.dispatching, policy.last_block = True, None
                    try:
                        policy.guard(name_, params)
                    finally:
                        policy.dispatching = False
                    decision, policy.last_block = policy.last_block, None
                label = command
            elif tool == "finish":
                final = str(_value(span, "answer") or answer)
                decision = policy.before_finish(final)
                label = f"finish: {final[:160]!r}"
            else:
                decision = policy.before_tool(tool, args)
                label = f"{tool}: {str(args.get('clarification_request', args))[:160]!r}"
                if tool == ASK_USER:
                    pending_question = str(args.get("clarification_request", ""))
            replayed = decision.source_policy if decision is not None else None
            if recorded or replayed:
                result.steps.append(ReplayedStep(
                    task_id=task_id, attempt=attempt,
                    step_index=attrs.get("step_index"), tool=str(tool),
                    label=label[:200], recorded=recorded, replayed=replayed,
                    rewrite=(decision.rewrite or "") if decision is not None else "",
                ))
            # Record what actually happened: a step blocked in the recording never ran.
            if recorded:
                continue
            if executed is not None:
                policy.record_command(executed[0], executed[1], executed[3], executed[2])
        policy.observe_answer(answer)
    return result


def replay_experiment(
    store: Any,
    experiment_id: str,
    table: DecisionTable,
    facts: ContractFacts,
    *,
    task_ids: Optional[list[str]] = None,
    live: bool = False,
) -> ReplayReport:
    """Replay every (or the listed tasks') attempt of an experiment."""
    from fastworkflow.distillation_compare import attempt_passed

    graded = {
        (str(r.get("task_id")), int(r.get("attempt") or 0)): attempt_passed(r)
        for r in store.experiment_attempt_rows(experiment_id)
    }
    report = ReplayReport(experiment_id=experiment_id, table_version=table.version, live=live)
    for task_id, attempt in sorted(dtrace.attempts_in_experiment(store, experiment_id)):
        if task_ids and str(task_id) not in {str(t) for t in task_ids}:
            continue
        report.attempts.append(replay_attempt(
            store, experiment_id, str(task_id), int(attempt), table, facts,
            live=live, passed=graded.get((str(task_id), int(attempt))),
        ))
    return report


def format_report(report: ReplayReport) -> str:
    """The report as text: every intervention, then the totals."""
    lines = [
        f"table {report.table_version} on {report.experiment_id} "
        f"({'live' if report.live else 'recorded'} intent)",
        "",
    ]
    for item in report.attempts:
        outcome = {True: "passed", False: "failed", None: "ungraded"}[item.passed]
        lines.append(f"=== task {item.task_id} attempt {item.attempt} ({outcome})")
        if item.error:
            lines.append(f"  skipped: {item.error}")
        for step in item.steps:
            lines.append(f"  s{step.step_index} [{step.mark}] {step.label}")
            lines.append(
                f"        recorded: {step.recorded or '-'}   "
                f"replayed: {step.replayed or '-'}"
            )
            if step.rewrite:
                lines.append(f"        says: {step.rewrite[:300]!r}")
        lines.append("")
    steps = [s for a in report.attempts for s in a.steps]
    lines.append(
        f"blocks recorded {sum(1 for s in steps if s.recorded)}, "
        f"replayed {sum(1 for s in steps if s.replayed)} "
        f"({sum(1 for s in steps if s.mark == 'LIFTED')} lifted, "
        f"{sum(1 for s in steps if s.mark == 'ADDED')} added); "
        f"replayed blocks on passing attempts: {report.fires_on(True)}, "
        f"on failing attempts: {report.fires_on(False)}"
    )
    return "\n".join(lines)
