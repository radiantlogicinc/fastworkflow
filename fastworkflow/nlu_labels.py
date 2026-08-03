"""Reserved NLU labels, and which pipeline stage each one belongs to.

fastWorkflow trains one intent classifier per command context. Its label space is
made of real command names plus a small number of *reserved* labels that do not
name any command. Historically a single reserved label, ``wildcard``, was used
for two unrelated jobs in two different NLU stages. The consequence was that the
escalation classifier was trained on parameter-value literals such as
``"france"`` — i.e. it was taught that "france" means "escalate to my parent".
This module exists so the two jobs have two names.

It is deliberately dependency-light (standard library only, no
``import fastworkflow``) so that both sides can import it:

- the trainer, ``fastworkflow.model_pipeline_training``, which assigns the labels
- the runtime, ``fastworkflow._workflows.command_metadata_extraction`` —
  ``intent_detection`` (which consumes predictions) and the ``wildcard`` command
  (which supplies the placeholder utterances)

The trainer cannot simply import the ``wildcard`` command module for these
constants: that is circular. ``wildcard.py`` imports ``intent_detection``, which
imports ``model_pipeline_training`` for ``CommandRouter``.

Stage ownership
---------------

``WILDCARD_LABEL`` (``"wildcard"``) belongs to **INTENT_DETECTION**. It is an
*escalation signal*. Training assigns it every ancestor-context utterance that is
not also valid in the local context, so a ``wildcard`` prediction means "the user
is asking for something this context does not offer; try my parent". At runtime
it resolves to ``command_name=None``, which drives the parent-chain walk in the
CME ``wildcard`` command and reaches ``ErrorCorrection/you_misunderstood`` only
once every ancestor has declined.

``PARAMETER_VALUE_LABEL`` (``"parameter_value"``) belongs to
**PARAMETER_EXTRACTION**. It is a *bare-value catcher*: the class that owns the
contentless literals a user types when answering a prompt (``"3"``, ``"france"``,
``"id=3636"``). Predicted during INTENT_DETECTION it means "this is a value, not
a local command". Like the wildcard label it resolves to ``command_name=None``,
but it carries no evidence that any ancestor can serve the utterance, so it is
*not* an escalation signal — see ``ESCALATION_LABELS`` below.

Neither label names a command, so neither may ever be looked up in a command
name map. That is what ``NON_ROUTABLE_LABELS`` is for.
"""

WILDCARD_LABEL = "wildcard"
PARAMETER_VALUE_LABEL = "parameter_value"

# The bare-value literals that make up the PARAMETER_VALUE_LABEL class. They are
# deliberately contentless: they exist to catch an utterance that is a value
# rather than a command. They live here rather than on the wildcard command's
# Signature so the trainer can label them independently of the escalation class.
PARAMETER_VALUE_PLACEHOLDERS: list[str] = [
    "3",
    "france",
    "16.7,.002",
    "John Doe, 56, 281-995-6423",
    "/path/to/my/object",
    "id=3636",
    "25.73 and Howard St",
]

# Labels that are never a command name. A prediction of one of these must resolve
# to `command_name=None`, and must never be displayed to a user as a choosable
# command.
NON_ROUTABLE_LABELS: frozenset[str] = frozenset({
    WILDCARD_LABEL,
    PARAMETER_VALUE_LABEL,
})

# The subset of NON_ROUTABLE_LABELS that means "an ancestor context may be able
# to serve this utterance". PARAMETER_VALUE_LABEL is deliberately excluded: a
# bare value says nothing about whether a parent context can help.
ESCALATION_LABELS: frozenset[str] = frozenset({
    WILDCARD_LABEL,
})


def label_of(prediction: str) -> str:
    """Return the bare label of a possibly fully-qualified prediction.

    Classifier labels are fully qualified (``Context/command``) for real
    commands and unqualified for the reserved labels; callers compare on the
    bare name.
    """
    return prediction.split("/")[-1]


def is_non_routable(prediction: str) -> bool:
    """True if *prediction* is a reserved label rather than a command name."""
    return label_of(prediction) in NON_ROUTABLE_LABELS


def is_escalation(prediction: str) -> bool:
    """True if *prediction* asks the runtime to try an ancestor context."""
    return label_of(prediction) in ESCALATION_LABELS
