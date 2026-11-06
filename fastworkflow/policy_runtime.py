"""The policy, as one agent runs it over one conversation.

``AgentPolicy`` is what a ReAct agent holds when its workflow has a policy
table (``fastworkflow.policy_loader.attach_policy``). It owns the conversation's
``PolicyState`` and is consulted at four points, each where the fact it needs
first exists:

* ``guard`` - from the command executor, after intent detection and parameter
  extraction and before the command runs (``BEFORE_SIDE_EFFECT``). Only there
  does the command exist in canonical form: its real name and validated
  parameters, whatever wording the agent used. Only the agent's own model
  steps are checked (``dispatching``): a command the user typed, or one a
  skill's code runs, is not the model's choice.
* ``before_tool`` - from the loop, before a non-command tool runs (a question to
  the user; ``AFTER_OBSERVATION``).
* ``before_finish`` - from the loop, when the agent picks ``finish`` and its
  answer has been written (``BEFORE_FINISH``).
* the ``observe_*`` / ``record_command`` calls - from the turn entry, the
  ``ask_user`` reply path and the command tool, so the rows see what happened.

Every decision is delivered as an ordinary observation the agent reads; nothing
here raises into a turn. Two valves bound a rule that is wrong about the
situation: a rule blocks the same kind of step at most ``MAX_BLOCKS_PER_RULE``
times per conversation, and a finished answer is pushed back at most
``MAX_FINISH_BLOCKS`` times per turn, and only after progress (a new write or a
new user reply) since the previous push.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from fastworkflow.policy_decision import (
    ASK_USER,
    USER_MESSAGE,
    USER_UTTERANCES,
    AfterObservationInput,
    BeforeFinishInput,
    BeforeSideEffectInput,
    CommandRecord,
    ContractFacts,
    PolicyDecision,
    PolicyDecisionPoint,
    PolicyState,
)
from fastworkflow.utils.logging import logger


class AgentPolicy:
    """One conversation's policy: its state, and the four decision points."""

    MAX_BLOCKS_PER_RULE = 2
    MAX_FINISH_BLOCKS = 3

    def __init__(
        self,
        point: PolicyDecisionPoint,
        facts: Optional[ContractFacts] = None,
        host: Any = None,
        *,
        reads_with_model: bool = True,
    ) -> None:
        self.point = point
        #: False in an offline replay with no model calls: intent is set by the
        #: caller from the recording, and consent is taken as given.
        self.reads_with_model = reads_with_model
        self.facts = facts or ContractFacts()
        #: The WorkflowExecutionContext, read for the user's message and the
        #: conversation; None in tests that drive the loop alone.
        self.host = host
        self.state = PolicyState()
        self._extractor = None
        self._judge = None
        # Per turn (reset by observe_turn): finish pushes, and what had
        # happened at the last one.
        self.finish_blocks = 0
        self.finish_mark: Optional[tuple[int, int]] = None
        # Set by the loop while a model step's command is being dispatched;
        # the guard ignores every other command.
        self.dispatching = False
        # The guard's block for the command being dispatched, read back by the
        # loop for the step's trace.
        self.last_block: Optional[PolicyDecision] = None
        # The question judge's reading for the step being decided.
        self.last_judgement = None

    @property
    def enabled(self) -> bool:
        return self.point.enabled

    # ------------------------------------------------------------------
    # What happened
    # ------------------------------------------------------------------

    def observe_turn(self, message: str) -> None:
        """A user message opened a turn: record it, and read what it asks for."""
        self.finish_blocks = 0
        self.finish_mark = None
        text = str(message or "")
        if not text.strip():
            return
        consents = self._read_intent(text)
        self.state.commands_run.append(
            CommandRecord(name=USER_MESSAGE, args={"consents": consents}, response=text)
        )

    def observe_reply(self, question: str, answer: str) -> None:
        """The user answered the agent's question (``ask_user``)."""
        question = str(question or "")
        consents = self._read_intent(str(answer or ""), asked=question)
        self.state.commands_run.append(
            CommandRecord(
                name=ASK_USER,
                args={"clarification_request": question, "consents": consents},
                response=str(answer or ""),
            )
        )

    def observe_answer(self, answer: str) -> None:
        """The agent's answer ended a run; the next user message replies to it."""
        if answer and answer.strip():
            self.state.dialogue.append(("assistant", answer))

    def record_command(
        self, name: str, args: Optional[dict], response: str, success: bool
    ) -> None:
        """A command ran. ``success`` is what the command reported; the
        workflow's ``failure_pattern`` can still mark its text as a failure."""
        text = str(response or "")
        self.state.commands_run.append(
            CommandRecord(
                name=str(name or ""),
                args=dict(args or {}),
                response=text,
                success=bool(success) and not self.facts.reports_failure(text),
            )
        )

    def _read_intent(self, text: str, asked: Optional[str] = None) -> bool:
        """Add the message to the dialogue and update the requests; the consent."""
        from fastworkflow import policy_intent

        if asked:
            self.state.dialogue.append((policy_intent.ASSISTANT, asked))
        self.state.dialogue.append((policy_intent.USER, text))
        if not self.reads_with_model:
            return True
        if not self.facts.intent_operations:
            return False
        if self._extractor is None:
            self._extractor = policy_intent.IntentExtractor(self.facts.intent_operations)
        tracked, consents = self._extractor.read(
            self.state.dialogue, policy_intent.tracked_from_dicts(self.state.requests)
        )
        self.state.requests = [dataclasses.asdict(r) for r in tracked]
        self.state.intent = list(policy_intent.for_rules(tracked))
        return consents

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def _common(self) -> dict:
        raw, conversation = "", []
        host = self.host
        if host is not None:
            try:
                workflow = host.get_active_workflow()
                raw = str((workflow.context or {}).get("raw_user_message", "") or "")
            except Exception:  # noqa: BLE001 - policy must never fail a turn
                raw = ""
        conversation = [text for who, text in self.state.dialogue if who == "user"]
        commands = tuple(self.state.commands_run)
        return {
            "facts": self.facts,
            "raw_user_message": raw,
            "conversation": tuple(conversation),
            "commands_run": commands,
            "observations": tuple(c.response for c in commands if c.response),
            "intent": tuple(self.state.intent),
        }

    def guard(self, command_name: str, params: Optional[dict]) -> Optional[str]:
        """The executor's call point: the rewrite to block with, or None to run.

        Never raises: a guard failure lets the command through. ``last_block``
        is set on a block and never cleared here: one model step can make a
        nested dispatch (the framework's own abort), and that check must not
        erase the step's block. The caller clears it before dispatching.
        """
        if not (self.enabled and self.dispatching):
            return None
        try:
            decision = self.point.decide(
                BeforeSideEffectInput(
                    **self._common(),
                    command_name=str(command_name or ""),
                    command_args=dict(params or {}),
                )
            )
            decision = self._valve(decision, "execute_workflow_query", command_name)
        except Exception as exc:  # noqa: BLE001 - policy must never fail a turn
            logger.warning(f"Policy guard skipped: {type(exc).__name__}: {exc}")
            return None
        if decision is None:
            return None
        self.last_block = decision
        return decision.rewrite

    def before_tool(self, tool_name: str, tool_args: Optional[dict]) -> Optional[PolicyDecision]:
        """A decision on a non-command tool the agent chose, or None."""
        self.last_judgement = None
        if not self.enabled:
            return None
        try:
            inputs = AfterObservationInput(
                **self._common(),
                pending_tool=str(tool_name or ""),
                pending_args=dict(tool_args or {}),
            )
            if tool_name == ASK_USER:
                inputs = self._judged(inputs)
            return self._valve(self.point.decide(inputs), tool_name, "")
        except Exception as exc:  # noqa: BLE001 - policy must never fail a turn
            logger.warning(f"Policy check skipped: {type(exc).__name__}: {exc}")
            return None

    def _judged(self, inputs: AfterObservationInput) -> AfterObservationInput:
        """The input with the question judge's reading, when the workflow asks for one."""
        if not (self.facts.judge_questions and self.reads_with_model):
            return inputs
        from fastworkflow import policy_intent

        if self._judge is None:
            self._judge = policy_intent.QuestionJudge()
        lookups = self.facts.lookup_hints(inputs) if self.facts.lookup_hints else ""
        judgement = self._judge.judge(
            str((inputs.pending_args or {}).get("clarification_request", "")),
            earlier=policy_intent.question_history(inputs.commands_run),
            lookups=lookups or "",
        )
        self.last_judgement = judgement
        if judgement is None:
            return inputs
        return dataclasses.replace(inputs, question_judgement=judgement)

    def before_finish(self, answer: str) -> Optional[PolicyDecision]:
        """A decision on the answer the agent is about to give, or None.

        Bounded per turn: at most ``MAX_FINISH_BLOCKS`` pushes, and a push only
        if a write or a user reply happened since the previous one. With
        neither, the rule is one the agent cannot satisfy, and pushing again
        would only spin the loop.
        """
        if not self.enabled or not (answer or "").strip():
            return None
        if self.finish_blocks >= self.MAX_FINISH_BLOCKS:
            return None
        try:
            decision = self.point.decide(BeforeFinishInput(**self._common(), answer=answer))
        except Exception as exc:  # noqa: BLE001 - policy must never fail a turn
            logger.warning(f"Policy finish check skipped: {type(exc).__name__}: {exc}")
            return None
        if decision is None:
            return None
        mark = (self._writes_done(), self._utterances_seen())
        if self.finish_mark == mark:
            return None
        self.finish_mark = mark
        self.finish_blocks += 1
        return decision

    def _writes_done(self) -> int:
        return sum(
            1 for c in self.state.commands_run if c.success and self.facts.is_write(c.name)
        )

    def _utterances_seen(self) -> int:
        return sum(1 for c in self.state.commands_run if c.name in USER_UTTERANCES)

    def _valve(self, decision, tool_name: str, command_name: str):
        """The decision, unless this rule has already blocked this step twice.

        If the agent comes back a third time it has read the rewrite twice and
        still judges the step necessary; a rule that is wrong about the
        situation would otherwise block it until the step cap.
        """
        if decision is None:
            return None
        key = f"{decision.source_policy}|{tool_name}|{command_name or ''}"
        counts = self.state.block_counts
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > self.MAX_BLOCKS_PER_RULE:
            logger.warning(
                f"Policy {decision.source_policy} has blocked "
                f"{command_name or tool_name} {self.MAX_BLOCKS_PER_RULE} times; "
                "letting it through"
            )
            return None
        return decision

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    def describe_intent(self) -> Optional[str]:
        from fastworkflow.policy_intent import describe

        return describe(tuple(self.state.intent))

    def describe_requests(self) -> Optional[str]:
        from fastworkflow.policy_intent import describe_tracked, tracked_from_dicts

        return describe_tracked(tracked_from_dicts(self.state.requests))

    # ------------------------------------------------------------------
    # Persistence (a session resumed in another process)
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        """Plain JSON data (a command's parameters may hold any value; they are
        kept as their text, which is what the rows compare anyway)."""
        import json

        data = self.state.to_dict()
        data["finish_blocks"] = self.finish_blocks
        data["finish_mark"] = list(self.finish_mark) if self.finish_mark else None
        return json.loads(json.dumps(data, default=str))

    def import_state(self, data: Optional[dict]) -> None:
        data = dict(data or {})
        self.finish_blocks = int(data.pop("finish_blocks", 0) or 0)
        mark = data.pop("finish_mark", None)
        self.finish_mark = tuple(mark) if mark else None
        self.state = PolicyState.from_dict(data)


def policy_of(agent: Any) -> Optional[AgentPolicy]:
    """The agent's policy when one is enabled, else None."""
    policy = getattr(agent, "policy", None)
    return policy if isinstance(policy, AgentPolicy) and policy.enabled else None
