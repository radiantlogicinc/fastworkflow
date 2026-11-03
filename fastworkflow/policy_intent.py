"""Model readings the policy rows take as inputs: intent, consent, questions.

Most policy rows need nothing but the trajectory: "was every record read before
this write" is a set difference over commands already run. A few cannot be
decided that way, because the fact they turn on is in the user's sentences, not
in any tool output -- "the user asked to change one part of a record, and the
agent is about to delete the whole record" needs to know which was asked for.

``IntentExtractor`` reads those sentences. It makes one model call per user
message, answering only in the operation vocabulary the workflow declares
(``ContractFacts.intent_operations``), so its output is always something the
rules can check against the commands that would satisfy it. Each call sees the
conversation so far (both sides' messages, no tool output) and the current
request list, and returns that list updated: a request can become due, turn
from a question into a request, or be withdrawn. It also reports whether the
message gives explicit consent to what the assistant last proposed, which only
makes sense read against the assistant's message.

``QuestionJudge`` reads a question the agent is about to ask the user, against
the earlier questions and what happened after each, and says whether it repeats
one with nothing new or asks for something the agent could look up itself.

The rules stay deterministic: given the same readings and the same trajectory
they decide the same way. The model's judgement is confined to reading.

Nothing here may fail a turn. A call that errors, times out, or returns
something unparseable yields no reading, and every rule that reads one is
written so that "no reading" means "nothing to enforce".

The model is ``LLM_POLICY_INTENT`` (key ``LITELLM_API_KEY_POLICY_INTENT``) when
set, otherwise the agent's own model, so it works with no extra configuration.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence

import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
import pydantic

import fastworkflow
from fastworkflow.policy_decision import (
    ASK_USER,
    USER_MESSAGE,
    QuestionJudgement,
    RequestedOperation,
)
from fastworkflow.utils import dspy_utils
from fastworkflow.utils.logging import logger

#: Seconds a policy model call may take before it is abandoned. A policy that
#: cannot answer in time enforces nothing; a slow one stalls the user's turn.
POLICY_TIMEOUT_S = 25

#: Longest a single message is shown to the reader. Answers can carry long
#: tables; the reader needs what was asked and said, not every row.
_MESSAGE_LIMIT = 2000

#: Speaker labels in the dialogue the reader sees.
USER = "user"
ASSISTANT = "assistant"


class _Request(pydantic.BaseModel):
    id: str = pydantic.Field(
        default="",
        description="The id it has in requests_so_far; empty for a new request.",
    )
    op: str = pydantic.Field(description="One of the operation names given.")
    scope: Literal["all", "specific", "unspecified"] = pydantic.Field(
        default="unspecified",
        description=(
            "'all' when the user asks for it on every record of a kind "
            '("all my open tickets"); \'specific\' when they name or describe '
            "particular ones; 'unspecified' otherwise."
        ),
    )
    status: Literal["firm", "fallback", "asks_if_possible", "withdrawn"] = (
        pydantic.Field(
            default="firm",
            description=(
                "'firm': the user wants it done. 'fallback': they want it only "
                "if something else they asked for is not possible, and the "
                "conversation has not yet shown that it is not. "
                "'asks_if_possible': they only asked whether it can be done "
                '("is there any way to...?") and have not asked for it to be '
                "done. 'withdrawn': they no longer want it - they said so, chose "
                "something else instead, or accepted the assistant's explanation "
                "that it cannot be done."
            ),
        )
    )
    detail: str = pydantic.Field(
        default="",
        description="A few words on what it applies to, in the user's terms.",
    )


class IntentExtractionSignature(dspy.Signature):
    """Keep the list of what a user has asked an assistant to do up to date
    after the user's newest message.

    You get the conversation so far (the user's and the assistant's messages,
    oldest first; the last one is the user's newest message), the only
    operation names you may use, and the requests recorded so far. Return the
    requests as they stand now: every request in requests_so_far, with its id,
    and any new ones (empty id).

    Record a request only if the user asks for it - not if they merely mention
    it or say they do NOT want it. A message that only supplies information (an
    identifier, a name, "yes") adds no request. Never drop a recorded request;
    keep each one's id, and change it only when the conversation shows a
    change:
    - correct its op or detail when the conversation shows it was misread -
      for example, what the user called parts of a record turn out to be whole
      records. Keep its scope: a request for 'all' stays 'all' unless the user
      themselves narrows it, and naming or confirming one record does not
      narrow it;
    - a 'fallback' becomes 'firm' once the assistant has said the thing it
      depends on is not possible and the user has not asked for something else;
    - an 'asks_if_possible' becomes 'firm' once the user, told it can be done,
      asks for it;
    - any request becomes 'withdrawn' when the user says they no longer want
      it, chooses something else instead, or accepts the assistant's
      explanation that it cannot be done.

    Separately, say whether the user's newest message gives explicit consent to
    carry out the action the assistant proposed in the message right before
    it. A bare "yes", "go ahead" or "please do" in reply to a proposal is
    consent. Supplying requested details, asking a question back, or hesitating
    is not. If the assistant's last message proposed nothing, there is no
    proposal to consent to."""

    conversation: str = dspy.InputField(
        desc='JSON list, oldest first: {"from": "user" | "assistant", "text": ...}.'
    )
    operations: str = dspy.InputField(
        desc="JSON object: the only operation names you may use -> what each means."
    )
    requests_so_far: str = dspy.InputField(
        desc="JSON list of the requests recorded so far: {id, op, scope, status, detail}."
    )
    requests: list[_Request] = dspy.OutputField(
        desc="Every recorded request (with its id) and any new ones, as they stand now."
    )
    consents_to_agent_proposal: bool = dspy.OutputField()


@dataclass(frozen=True)
class TrackedRequest:
    """One request as the reader keeps it: a stable id and its current status.

    Rules never see this directly - ``for_rules`` turns the list into the
    ``RequestedOperation``s they read.
    """

    id: str
    op: str
    scope: str = "unspecified"
    status: str = "firm"  # firm | fallback | asks_if_possible | withdrawn
    detail: str = ""


_STATUSES = ("firm", "fallback", "asks_if_possible", "withdrawn")


def for_rules(requests: Sequence[TrackedRequest]) -> tuple[RequestedOperation, ...]:
    """What the rules enforce: firm requests, and fallbacks (as conditional).

    A request the user only asked about, or withdrew, is not required by any
    rule, so it is left out rather than passed on with a flag every rule would
    have to learn to ignore.
    """
    return tuple(
        RequestedOperation(
            op=r.op,
            scope=r.scope,
            conditional=r.status == "fallback",
            detail=r.detail,
        )
        for r in requests
        if r.status in ("firm", "fallback")
    )


def policy_lm():
    """The model the policy readers use, with a hard time limit and no retries.

    ``LLM_POLICY_INTENT`` when set, otherwise the agent's own model.
    """
    limits = {"timeout": POLICY_TIMEOUT_S, "num_retries": 0}
    if fastworkflow.get_env_var("LLM_POLICY_INTENT", default=""):
        return dspy_utils.get_lm(
            "LLM_POLICY_INTENT", "LITELLM_API_KEY_POLICY_INTENT", **limits
        )
    return dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT", **limits)


def _clip(text: str, limit: int = _MESSAGE_LIMIT) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


class IntentExtractor:
    """One model call per user message, answered in the workflow's vocabulary.

    Each call sees the whole conversation (messages only - no tool calls or
    results) and the current request list, and returns the list as it stands
    now, so a request can change status instead of only ever being added.

    Results are memoised on the exact inputs, so a message is never read twice
    within one process even with the provider cache disabled.
    """

    def __init__(self, operations: Mapping[str, str], lm=None) -> None:
        self.operations = dict(operations)
        self._vocabulary = json.dumps(self.operations, indent=1)
        self._memo: dict[tuple[str, str], tuple] = {}
        self._predict = dspy.Predict(IntentExtractionSignature)
        self._lm = lm
        self.calls = 0
        self.failures = 0

    def read(
        self,
        conversation: Sequence[tuple[str, str]],
        current: Sequence[TrackedRequest] = (),
    ) -> tuple[tuple[TrackedRequest, ...], bool]:
        """(the request list as it stands now, whether the newest message consents).

        ``conversation`` is (speaker, text) pairs, oldest first, speaker
        ``USER`` or ``ASSISTANT``; the last pair is the user's newest message.

        Never raises. Any failure is logged and leaves the list as it was, with
        no consent - the conservative reading for every rule that uses it. A
        recorded request the model leaves out is kept unchanged: a request ends
        only by being marked withdrawn, never by omission.
        """
        current = tuple(current)
        dialogue = [
            {"from": who, "text": _clip(text)}
            for who, text in conversation
            if (text or "").strip()
        ]
        if not dialogue or dialogue[-1]["from"] != USER or not self.operations:
            return current, False
        conversation_text = json.dumps(dialogue, ensure_ascii=False)
        current_text = json.dumps(
            [dataclasses.asdict(r) for r in current], ensure_ascii=False
        )
        key = (conversation_text, current_text)
        if key in self._memo:
            return self._memo[key]
        try:
            self.calls += 1
            if self._lm is None:
                self._lm = policy_lm()
            with dspy.context(lm=self._lm):
                out = self._predict(
                    conversation=conversation_text,
                    operations=self._vocabulary,
                    requests_so_far=current_text,
                )
            result = (
                apply_requests(current, out.requests or [], self.operations),
                bool(out.consents_to_agent_proposal),
            )
        except Exception as exc:  # noqa: BLE001 - reading intent must not fail a turn
            self.failures += 1
            logger.warning(
                f"Intent extraction failed ({type(exc).__name__}: {exc}); "
                "keeping the request list as it was"
            )
            result = (current, False)
        self._memo[key] = result
        return result


def apply_requests(current, returned, operations) -> tuple[TrackedRequest, ...]:
    """The new list: recorded requests updated in place, new ones appended.

    Ids are ours, not the model's: a returned id that is not in ``current`` is
    a new request, and gets the next id. Operations outside the vocabulary are
    dropped and unknown statuses read as firm.
    """
    by_id = {r.id: r for r in current}
    updated = dict(by_id)
    added: list[TrackedRequest] = []
    next_n = len(current) + 1
    for r in returned:
        op = _field(r, "op", "")
        if op not in operations:
            continue
        status = _field(r, "status", "firm")
        status = status if status in _STATUSES else "firm"
        rid = str(_field(r, "id", "") or "")
        fields = dict(
            op=op,
            scope=_field(r, "scope", "unspecified") or "unspecified",
            status=status,
            detail=str(_field(r, "detail", "") or ""),
        )
        if rid in by_id:
            updated[rid] = TrackedRequest(id=rid, **fields)
        else:
            added.append(TrackedRequest(id=f"R{next_n}", **fields))
            next_n += 1
    return tuple(updated[r.id] for r in current) + tuple(added)


def _field(obj, name: str, default):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


class QuestionJudgeSignature(dspy.Signature):
    """Check a question an assistant is about to send to the user, before it
    is sent.

    You get the question; the questions the assistant already asked in this
    conversation, each with the user's answer and what the assistant did
    afterwards; and what the assistant could still look up for itself.

    repeats_without_new_information: true only if BOTH hold: (a) the question
    asks for essentially the same thing as an earlier question, and (b)
    nothing since gives a reason to ask for it again. A reason is: something
    the assistant did with the answer FAILED or did not settle it (e.g. a
    lookup with the user's answer returned "not found"), the answer turned out
    wrong or incomplete, the user changed what they want, or the user said they
    were unsure or asked for help and the assistant now offers concrete options
    to choose from. The user replying WITHOUT giving what was asked - not
    knowing it, or talking about something else - is NOT a reason: asking the
    same thing again after that is a repeat. Presenting findings and asking the
    user to confirm a proposed action is never a repeat.

    asks_for_what_agent_can_look_up: true if the question asks the user to
    supply, identify, or verify something the assistant could find out itself
    from what is listed as still available to look up. That includes asking
    which record holds something, asking for a record's identifier, and asking
    the user to confirm that a particular record is the right one, while those
    records have not been read. False for asking whether to go ahead with an
    action, for a choice by the user's own preference, and for anything only
    the user knows (a new value they want, their own contact details).

    Write the reasoning first, checking each condition, then the verdicts."""

    question: str = dspy.InputField()
    earlier_questions: str = dspy.InputField(
        desc=(
            "JSON list, oldest first: {question, user_answer, "
            "assistant_did_since}. assistant_did_since runs up to the next "
            "question; everything after an earlier question counts as "
            "happening since it."
        )
    )
    available_to_look_up: str = dspy.InputField(
        desc="What the assistant could still look up itself; empty if nothing."
    )
    reasoning: str = dspy.OutputField(
        desc="Two or three sentences, checking the conditions above in turn."
    )
    repeats_without_new_information: bool = dspy.OutputField()
    asks_for_what_agent_can_look_up: bool = dspy.OutputField()
    reason: str = dspy.OutputField(desc="One sentence.")


class QuestionJudge:
    """One model call per question the agent tries to ask, when there is
    something to judge it against.

    Skipped - no call made - when the agent has asked nothing before and there
    is nothing left for it to look up: with neither, there is nothing a
    question could repeat or duplicate. Shares the intent reader's model and
    failure contract: a failed call yields no judgement, and rules that read a
    judgement do nothing without one.
    """

    def __init__(self, lm=None) -> None:
        self._memo: dict[tuple[str, str, str], Optional[QuestionJudgement]] = {}
        self._predict = dspy.Predict(QuestionJudgeSignature)
        self._lm = lm
        self.calls = 0
        self.failures = 0

    def judge(
        self, question: str, *, earlier: list[dict], lookups: str
    ) -> Optional[QuestionJudgement]:
        question = (question or "").strip()
        if not question or (not earlier and not lookups):
            return None
        earlier_text = json.dumps(earlier, ensure_ascii=False)
        key = (question, earlier_text, lookups or "")
        if key in self._memo:
            return self._memo[key]
        try:
            self.calls += 1
            if self._lm is None:
                self._lm = policy_lm()
            with dspy.context(lm=self._lm):
                out = self._predict(
                    question=question,
                    earlier_questions=earlier_text,
                    available_to_look_up=lookups or "",
                )
            result: Optional[QuestionJudgement] = QuestionJudgement(
                repeats_without_new_information=bool(
                    out.repeats_without_new_information
                ),
                asks_for_what_agent_can_look_up=bool(
                    out.asks_for_what_agent_can_look_up
                ),
                reason=str(out.reason or ""),
            )
        except Exception as exc:  # noqa: BLE001 - judging must not fail a turn
            self.failures += 1
            logger.warning(
                f"Question judgement failed ({type(exc).__name__}: {exc}); "
                "letting the question through unjudged"
            )
            result = None
        self._memo[key] = result
        return result


def question_history(commands_run, *, limit: int = 300) -> list[dict]:
    """Each earlier question, its answer, and everything that happened after it.

    Built from the policy's event list (commands and user utterances in order),
    for the question judge: whether asking again is a repeat depends on what
    happened in between, not on the wording of the two questions.
    """
    history: list[dict] = []
    for record in commands_run:
        if record.name == ASK_USER:
            history.append(
                {
                    "question": _clip(
                        record.args.get("clarification_request", ""), limit
                    ),
                    "user_answer": _clip(record.response, limit),
                    "assistant_did_since": [],
                }
            )
        elif history:
            if record.name == USER_MESSAGE:
                event = f"user said: {_clip(record.response, limit)}"
            else:
                args = ", ".join(f"{k}={v}" for k, v in (record.args or {}).items())
                outcome = "ok" if record.success else "FAILED"
                event = (
                    f"{record.name}({_clip(args, limit)}) -> {outcome}: "
                    f"{_clip(record.response, limit)}"
                )
            history[-1]["assistant_did_since"].append(event)
    return history


def describe(intent: Sequence[RequestedOperation]) -> Optional[str]:
    """Human-readable intent, for a trace attribute.

    ``parse_described`` reads it back (the replay audit uses it to replay a run
    with the intent it recorded).
    """
    if not intent:
        return None
    return "; ".join(
        f"{r.op}{'?' if r.conditional else ''}"
        f"{'[all]' if r.scope == 'all' else ''}({r.detail})"
        for r in intent
    )


def parse_described(text: Optional[str]) -> tuple[RequestedOperation, ...]:
    """The inverse of ``describe``, as far as it records."""
    import re

    if not text:
        return ()
    return tuple(
        RequestedOperation(
            op=op,
            conditional=bool(q),
            scope="all" if all_ else "unspecified",
            detail=detail,
        )
        for op, q, all_, detail in re.findall(
            r"(\w+)(\?)?(\[all\])?\((.*?)\)(?:;\s*|$)", text
        )
    )


def describe_tracked(requests: Sequence[TrackedRequest]) -> Optional[str]:
    """The reader's full list with statuses, for a trace attribute."""
    if not requests:
        return None
    return "; ".join(
        f"{r.id}:{r.op}[{r.status}]{'[all]' if r.scope == 'all' else ''}({r.detail})"
        for r in requests
    )


def tracked_from_dicts(rows: Sequence[Mapping]) -> tuple[TrackedRequest, ...]:
    """``PolicyState.requests`` back into ``TrackedRequest``s."""
    return tuple(TrackedRequest(**dict(r)) for r in rows)
