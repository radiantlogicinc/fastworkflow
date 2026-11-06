# Writing a policy table

A table module exposes `TABLE` (a `DecisionTable`) and optionally `FACTS` (`ContractFacts`).
Put it at `<workflow>/_policy/table.py`; the workflow enforces it from then on.

## Facts: what the workflow declares

```python
from fastworkflow.policy_decision import ContractFacts

FACTS = ContractFacts(
    write_commands=frozenset({"close_ticket", "reassign_ticket"}),  # commands that change state
    id_patterns=(r"T-\d+",),          # identifier shapes; default: _commands/_split.py identifier_patterns
    enum_arguments={"close_ticket.resolution": frozenset({"fixed", "wont_fix"})},
    intent_operations={               # operations a user can ask for; enables the intent reader
        "close_ticket": "Close one or more tickets as resolved.",
        "reassign_ticket": "Move a ticket to another team or person.",
    },
    failure_pattern=None,             # regex for tools that report errors as successful text
    judge_questions=False,            # True runs the question judge before every ask_user
    lookup_hints=None,                # f(inputs) -> "what the agent could still look up", for the judge
)
```

Declare only what a predicate reads. Leave `intent_operations` empty unless a rule needs to
know what was asked: each entry costs one model call per user message.

## Phases and what each input carries

| phase | when | input | extra fields |
|---|---|---|---|
| `BEFORE_SIDE_EFFECT` | a command the model chose, after parameter extraction, before it runs | `BeforeSideEffectInput` | `command_name`, `command_args` (validated parameters), `id_arguments()` |
| `AFTER_OBSERVATION` | a non-command tool the model chose (a question to the user) | `AfterObservationInput` | `pending_tool`, `pending_args`, `question_judgement` |
| `BEFORE_FINISH` | the agent picked `finish` and its answer is written | `BeforeFinishInput` | `answer` |

Every input carries:

- `facts`, `raw_user_message`;
- `conversation`: the user's messages;
- `commands_run`: the event list below, oldest first;
- `observations`;
- `intent`: what the user asked for, as `RequestedOperation(op, scope, conditional, detail)`.

It also has these helpers:

- `ran(name)`: successful runs of a command;
- `last_user_utterance()`;
- `requested(op, conditional=)`;
- `observation_text()`.

`commands_run` holds `CommandRecord(name, args, response, success)` for:

- every command that ran, under its real name with its validated parameters;
- `"@user"` (`USER_MESSAGE`): a message that opened a turn; `response` is the text;
- `"ask_user"` (`ASK_USER`): a question and its answer. `args["clarification_request"]` is
  the question, `response` is the answer.

Both user records carry `args["consents"]`: whether the message explicitly agreed to what the
assistant proposed last. This is read by the intent model; it is always False without
`intent_operations`. Commands blocked by a rule never ran and are not in the list.

The history spans the whole conversation, not one turn.

## Rows

```python
from fastworkflow.policy_decision import DecisionPhase, DecisionTable, PolicyOutcome, PolicyRow

BEFORE_WRITE = frozenset({DecisionPhase.BEFORE_SIDE_EFFECT})


def _unread(inputs) -> list[str]:
    listed = {t for r in inputs.ran("list_tickets") for t in re.findall(r"T-\d+", r.response)}
    read = {r.args.get("ticket_id") for r in inputs.ran("get_ticket")}
    return sorted(listed - read)


TABLE = DecisionTable("helpdesk/1.0.0", [
    PolicyRow(
        row_id="H1-read-every-ticket-before-closing-one",
        phases=BEFORE_WRITE,
        outcome=PolicyOutcome.REWRITE,
        reason="a close while listed tickets are unread",
        source_insights=("E3", "P1"),             # taxonomy ids, for traceability
        predicate=lambda i: i.command_name == "close_ticket" and bool(_unread(i)),
        rewrite=lambda i: (f"Tickets {', '.join(_unread(i))} are not read yet. Call "
                           "get_ticket on each, then choose which to close."),
    ),
])
```

How rows behave:

- **Predicates must be pure and cheap.** They run on every step at their phase.
- **`rewrite`** is a string, or a function of the input that names what was wrong.
- **Order.** First match wins.
- **Required fields.** A version string and unique row ids are required.
- **`ALLOW`** marks an exemption: a row that matches to stop later rows from firing.
- **Failure handling.** A predicate that raises lets the step through, and the failure is
  logged.

## Testing a row

```python
from fastworkflow.policy_decision import BeforeSideEffectInput, CommandRecord

history = (CommandRecord("list_tickets", {}, "T-1, T-2"),
           CommandRecord("get_ticket", {"ticket_id": "T-1"}, '{"id": "T-1"}'))
inputs = BeforeSideEffectInput(facts=FACTS, commands_run=history,
                               observations=tuple(r.response for r in history),
                               command_name="close_ticket", command_args={"ticket_id": "T-1"})
assert TABLE.evaluate(inputs).source_policy.startswith("H1")
```

Build histories from the recorded trials the row came from: one where it must fire, and one
from a passing trial where it must stay silent. Then audit the table on whole runs with
`fastworkflow policy replay`.
