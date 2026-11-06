"""The planning and execution argument checkers agree, except where they differ on purpose (fix-48np.3).

Two checkers read an argument against the request:

- **planning**: ``planning_commands.checked_arguments`` (``check_arguments``, then ``_validate_arguments``:
  the hook, ``check_input``, ``_check_roles``), on the values ``extracted_arguments`` hands the typed call
  (the extractor's "not given" text read as no value, ``_given_or_none``);
- **execution**: ``argument_checks.check_extracted_values`` and ``ExtractionCheck.apply``, on the
  ``model_construct`` instance the LLM extractor gives an execution command (``ParameterExtraction._llm_extract``).

Every case of one table goes through both, on the same pydantic ``Input`` (a real planning command of
``tests/planning_tutorial_workflow``, or one of the small commands below, written into the temporary copy
for a field shape the tutorial lacks), and each checker's outcome for the field under test is reduced to one
verdict: accepted as given, accepted as another value (respelled, or replaced), refused, or not given.

The verdicts must be equal, except for a case that names one of the ``DIFFERENCES`` below: such a case must
show exactly the difference its entry describes (so the list stays truthful), and each entry must have a case.
A new disagreement fails the test with both verdicts. The fix is then one of two, chosen on purpose: align
the checkers (move the rule into the shared module, ``fastworkflow/utils/argument_checks.py``, with both paths
calling it, and its gate measured: fix-48np.2's notes), or add a documented entry to ``DIFFERENCES``, by an
explicit decision, with the measurement for the path whose behaviour differs.

Out of scope here, by design: a ``Literal`` choice, which the execution checker leaves to the type check of
``validate_parameters`` (restated in ``_literal_verdict``, a closure there), and the execution path's option
fields, which ``validate_parameters`` checks with ``check_arguments`` itself (one implementation: no drift).

The table: the strictness corpus of ``tests/test_typed_call_strictness.py`` (every case of it, each classified
in ``CORPUS_FIELDS``), the field-by-field cases of ``tests/test_extraction_argument_checks.py`` adapted to fields
both checkers read, and cases for each deliberate difference. No model is called.
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, get_args, get_origin

import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import planning_commands as pc
from fastworkflow import workflow_agent
from fastworkflow.utils import argument_checks
from fastworkflow.utils.signatures import INVALID_INT_VALUE, NOT_FOUND
from tests.test_planning_command_argument_screens import TRANSFER_LOAN
from tests.test_typed_call_strictness import ACCOUNTS, CORPUS

TUTORIAL = Path(__file__).resolve().parent / "planning_tutorial_workflow"

#: Planning commands for field shapes the tutorial lacks, written into the temporary copy only. Each takes
#: one shape, so a rule of another field (the planning number check reads every ``in_request`` int) does not
#: decide a case. Their execution twins are the same ``Input``s.
DRIFT_COMMANDS = {
    # an int both checkers read (in_request: the planning number check), beside a str holding digits
    "drift_rows": '''
from pydantic import BaseModel, Field


class Signature:
    """List rows of the books, for the drift test."""

    class Input(BaseModel):
        title: str = Field(default="", description="the book's title", json_schema_extra={"in_request": True})
        count: int = Field(default=0, description="how many rows", json_schema_extra={"in_request": True})


class PlanGenerator:
    def steps(self, input):
        return [f"list {input.count} rows of the book {input.title}"]
''',
    # an int whose default is a real number (the execution checker reads a default as "not given")
    "drift_page": '''
from pydantic import BaseModel, Field


class Signature:
    """Show a page of loans, for the drift test."""

    class Input(BaseModel):
        size: int = Field(default=5, description="rows on the page", json_schema_extra={"in_request": True})


class PlanGenerator:
    def steps(self, input):
        return [f"show {input.size} loans"]
''',
    # str fields both read against the request: a name, and a code whose description gives an example
    "drift_code": '''
from pydantic import BaseModel, Field


class Signature:
    """Label a member's shelf, for the drift test."""

    class Input(BaseModel):
        name: str = Field(description="a person's name", json_schema_extra={"in_request": True})
        code: str = Field(default="", description="a shelf code", examples=["XY-12"],
                          json_schema_extra={"in_request": True})


class PlanGenerator:
    def steps(self, input):
        return [f"find the member {input.name}", f"label the shelf {input.code}"]
''',
    # a value converted on purpose (in_request False), with an example
    "drift_converted": '''
from pydantic import BaseModel, Field


class Signature:
    """Set a loan's due date, for the drift test."""

    class Input(BaseModel):
        name: str = Field(default="", description="a person's name", json_schema_extra={"in_request": True})
        due: str = Field(default="", description="the due date, YYYY-MM-DD", examples=["2024-05-20"],
                         json_schema_extra={"in_request": False})


class PlanGenerator:
    def steps(self, input):
        return [f"set the due date of {input.name} to {input.due}"]
''',
    # a value converted into a declared pattern
    "drift_shelf": '''
from pydantic import BaseModel, Field


class Signature:
    """Move a loan to a shelf, for the drift test."""

    template = ["move the loan to the shelf"]   # the pattern refuses the placeholder inputs

    class Input(BaseModel):
        shelf: str = Field(description="the shelf code", examples=["#S1"], pattern=r"^#S[0-9]+$")


class PlanGenerator:
    def steps(self, input):
        return [f"move the loan to shelf {input.shelf}"]
''',
    # a choice: a Literal (the planning enum); its execution twins are ChoiceText's fields
    "drift_choice": '''
from typing import Literal

from pydantic import BaseModel, Field


class Signature:
    """Find a record of one kind, for the drift test."""

    class Input(BaseModel):
        kind: Literal["book", "member", "branch"] = Field(default="book", description="the kind of record")
        kind_prose: str = Field(default="book", description="the kind of record. One of: book, member, branch")


class PlanGenerator:
    def steps(self, input):
        return [f"find a {input.kind}"]
''',
    # the subject option
    "drift_subject": '''
from pydantic import BaseModel, Field


class Signature:
    """Show one record's books, for the drift test."""

    class Input(BaseModel):
        name: str = Field(description="the record's name", json_schema_extra={"in_request": "subject"})


class PlanGenerator:
    def steps(self, input):
        return [f"show the books of {input.name}"]
''',
    # a verify choice (the classifier's verdict is data the planning call is given; no model here)
    "drift_verify": '''
from typing import Literal

from pydantic import BaseModel, Field


class Signature:
    """Find a record of the kind the request means, for the drift test."""

    class Input(BaseModel):
        name: str = Field(default="", description="the name to look for", json_schema_extra={"in_request": True})
        kind: Literal["book", "member", "branch"] = Field(default="branch", description="the kind of record",
                                                          json_schema_extra={"verify": True})


class PlanGenerator:
    def steps(self, input):
        return [f"find the {input.kind} {input.name}"]
''',
}




class ChoiceText(BaseModel):
    """The execution twin of ``drift_choice.kind``: the same choice declared as an execution command's ``str``
    allowed values (``json_schema_extra`` enum, which a planning command's loader refuses: its choice is a
    ``Literal``)."""

    kind_text: str = Field(default="book", description="the kind of record",
                           json_schema_extra={"enum": ["book", "member", "branch"]})


# ---------------------------------------------------------------- verdicts

ACCEPTED = "accepted"
REFUSED = "refused"
NOT_GIVEN = "not given"


def _as(value) -> tuple:
    """Accepted as another value than the one given: respelled ("dune messiah" -> "Dune Messiah"), or
    replaced (4421 -> acct_4421, 20 -> 10)."""
    return ("accepted as", value)


def _shown(verdict) -> str:
    return f"accepted as {verdict[1]!r}" if isinstance(verdict, tuple) else verdict


# ---------------------------------------------------------------- the deliberate differences

@dataclass(frozen=True)
class Difference:
    """A way the checkers differ on purpose: what, why, the measurement behind it, and the verdict pair a
    case of it must show (``shows(planning, execution)``)."""

    what: str
    why: str
    measured: str
    shows: Callable[[Any, Any], bool]


def _replaced(verdict) -> bool:
    return isinstance(verdict, tuple)


#: The deliberate differences (fix-48np.2, owner decision 2026-11-06). An entry is removed, or one added, only by
#: an explicit decision, with the measurement for the path whose behaviour changes (fix-48np.2's notes, step 4).
DIFFERENCES: dict[str, Difference] = {
    "int-not-written": Difference(
        what="An int the request does not write, when it writes exactly one number (or several, one of them the "
             "field's default): planning replaces it with that number; execution refuses it (a value given) or "
             "leaves it not given (a value the extractor did not give).",
        why="On an execution command a replaced number is a silent wrong change to a command that acts; a refusal "
            "asks the extractor once more, then the user.",
        measured="fix-qjf4 (the app's extraction replay): replacing made wrong changes, so the execution rule "
                 "refuses; the planning rule is gated by the strictness corpus.",
        shows=lambda planning, execution: _replaced(planning) and execution in (REFUSED, NOT_GIVEN)),
    "absent-int": Difference(
        what="An int the extractor leaves out (None, its not-given number, or the default 0): planning refuses it "
             "when the request writes two or more numbers, none of them the default; execution never refuses an "
             "absent value. (An Optional[int] cannot carry the planning in_request option: option_params refuses "
             "it, so the planning side of this row is an in_request int with a plain default.)",
        why="An absent value is not a value the extractor gave; refusing it refuses a correct call.",
        measured="22 of 974 Haiku calls (2.3%) refused by the planning rule on an absent Optional[int] in the "
                 "app's extraction replay (fix-qjf4, tests/test_extraction_argument_checks.py).",
        shows=lambda planning, execution: planning == REFUSED and execution == NOT_GIVEN),
    "exemptions": Difference(
        what="Which fields are read against the request: planning checks a str field only when it carries a "
             "check option (in_request, opt in per field, design 2.46); execution checks every str field unless "
             "it is exempt (json_schema_extra in_request False, or a declared pattern: opt out), and gives an "
             "exempt field no description-example check.",
        why="A planning command's author declares its checks; an execution command's Input predates the options, "
            "so its checks are on by default and a converted value is exempted by the marker or a pattern.",
        measured="0 refused user-stated gold values on tau-bench retail (153), airline (105) and the retail set "
                 "(105); the airline JFK / 2024-05-20 false refusals motivated no example check on exempt fields.",
        shows=lambda planning, execution: planning == ACCEPTED and execution == REFUSED),
    "which-values": Difference(
        what="Which values are checked: execution checks only values the LLM extractor produced, and reads a value "
             "equal to the field's declared default as not given (the extractor is told to answer the default for "
             "a value the request leaves out); planning checks every argument of the call, a default included.",
        why="On the execution path a default is indistinguishable from \"not given\"; refusing it refuses a correct "
            "call. (Values from other extraction methods, stored parameters, the XML form or a direct action, "
            "never reach check_extracted_values: ParameterExtraction._llm_extract is its only caller.)",
        measured="Execution gate: 0 of 1,189 values refused on passing Haiku runs, 32 of 107 wrong Nemotron values "
                 "caught (fix-qjf4 replay).",
        shows=lambda planning, execution: planning != NOT_GIVEN and execution == NOT_GIVEN),
    "planning-only": Difference(
        what="Planning-only concepts, not imposed on execution commands: subjects (the in_request \"subject\" "
             "option, and a pronoun read as the request's one subject or refused), identifier expansion (a "
             "fragment of an identifier the request writes becomes the whole identifier, or is refused), kind "
             "selectors and the vocabulary word screens (a kind, field or command word of the workflow), verify "
             "verdicts, and roles (two inputs given one value; an optional str repeating another).",
        why="They read the workflow (its vocabulary, identifiers, subjects) or a classifier and decide what a "
            "planning call means; on an execution command they stay opt-in per field (the option fields "
            "validate_parameters checks with check_arguments itself), not checks of every extracted value.",
        measured="Owner decision 2026-11-06 (fix-48np.2): no failure measured that session came from having two "
                 "checkers; imposing these on execution commands is a behaviour change to measure first.",
        shows=lambda planning, execution: planning != execution and execution == ACCEPTED),
}


# ---------------------------------------------------------------- the case table

@dataclass(frozen=True)
class Case:
    id: str
    source: str               # where the case comes from
    command: str              # the planning command whose Input both checkers read
    request: str
    values: dict              # the extractor's values
    field: str                # the field whose verdicts are compared
    execution_field: str | None = None   # the field's execution twin (a choice declared another way) ...
    execution_model: type | None = None  # ... on this execution Input
    verdicts: dict | None = None         # the classifier's verify verdicts, planning only
    difference: str | None = None        # a DIFFERENCES key: the case shows that difference
    drift: str | None = None             # a disagreement found and not yet decided: xfail(strict)


#: Each strictness corpus case: the field compared, and the difference it shows (None: the checkers agree).
#: Every corpus case must be here, so a new one is classified when it is added.
CORPUS_FIELDS: dict[str, tuple[str, str | None]] = {
    "title": ("title", None),
    "title-case": ("title", None),
    "person": ("author", None),
    "two-roles": ("member", None),
    "pronoun-one-subject": ("author", "planning-only"),        # "it" is the request's one subject
    "enum": ("kind", None),
    "bools": ("title", None),
    "whole-identifier": ("title", "planning-only"),           # 4421 -> acct_4421
    "optional-placeholder": ("note", None),
    "braces": ("author", None),
    "angle": ("author", None),
    "name-number": ("author", None),
    "unspecified": ("author", None),
    "invented-example": ("author", None),
    "that-one": ("author", "planning-only"),                  # a pronoun, no single subject
    "it": ("author", "planning-only"),
    "own-name": ("author", "planning-only"),                  # vocabulary words
    "field-name": ("query", "planning-only"),
    "command-word": ("member", "planning-only"),
    "declared-choice": ("branch", "planning-only"),
    "kind-word": ("query", "planning-only"),
    "invented-subject": ("author", None),
    "not-in-request": ("title", None),
    "deictic": ("title", None),
    "identifier-fragment": ("title", "planning-only"),
    "role-conflict": ("branch", "planning-only"),             # branch is the input refused
    "role-conflict-unwritten": ("member", "planning-only"),
    "not-an-enum-value": ("kind", None),
    "not-given-text": ("title", None),
    "unknown-marker": ("title", None),
    "missing-required": ("author", None),
}

CORPUS_CASES = [Case(case_id, "strictness corpus", name, request, values, field=CORPUS_FIELDS[case_id][0],
                     difference=CORPUS_FIELDS[case_id][1])
                for case_id, name, request, values in CORPUS if case_id in CORPUS_FIELDS]

_EXEC = "extraction argument checks"

#: tests/test_extraction_argument_checks.py, each case on a field both checkers read.
EXECUTION_CASES = [
    Case("invented-name", _EXEC, "drift_code", "find the member Ann Lee", {"name": "Bob Jones"}, "name"),
    Case("near-miss-name", _EXEC, "drift_code", "find the member Ann Lee", {"name": "Ann Leigh"}, "name"),
    Case("request-spelling", _EXEC, "drift_code", "find the member Ann Lee", {"name": "ann lee."}, "name"),
    Case("dash-variant", _EXEC, "drift_code", "find the member Ann Lee–Smith", {"name": "Ann Lee-Smith"}, "name"),
    Case("int-written", _EXEC, "drift_rows", "list the loans of book b1 from row 10", {"count": 10}, "count"),
    Case("int-not-written", _EXEC, "drift_rows", "list the loans of book b1 from row 10", {"count": 20}, "count",
         difference="int-not-written"),
    Case("absent-int-two-numbers", _EXEC, "drift_rows", "list the loans of book b1 for the 2 or 3 members who asked",
         {"count": None}, "count", difference="absent-int"),
    Case("absent-int-sentinel", _EXEC, "drift_rows", "list the loans of book b1 for the 2 or 3 members who asked",
         {"count": INVALID_INT_VALUE}, "count", difference="absent-int"),
    Case("choice-outside", _EXEC, "drift_choice", "cancel it, the price dropped elsewhere",
         {"kind": "price dropped elsewhere"}, "kind", execution_field="kind_text", execution_model=ChoiceText),
    Case("choice-declared-spelling", _EXEC, "drift_choice", "find the member Ann", {"kind": "Member"}, "kind",
         execution_field="kind_text", execution_model=ChoiceText),
    Case("choice-member", _EXEC, "drift_choice", "find the member Ann", {"kind": "member"}, "kind",
         execution_field="kind_text", execution_model=ChoiceText),
    Case("pattern-converted", _EXEC, "drift_shelf", "put Ann Lee's loan on shelf S3", {"shelf": "#S3"}, "shelf"),
    Case("in-request-false-converted", _EXEC, "drift_converted", "add Ann Lee, due next Monday",
         {"due": "2026-11-09"}, "due"),
    Case("exempt-placeholder", _EXEC, "drift_converted", "add Ann Lee, due next Monday", {"due": "{due_date}"}, "due"),
    Case("description-example-copied", _EXEC, "drift_code", "add Ann Lee", {"name": "Ann Lee", "code": "XY-12"},
         "code"),
    Case("description-example-written", _EXEC, "drift_code", "add Ann Lee with code XY-12",
         {"name": "Ann Lee", "code": "XY-12"}, "code"),
    Case("converted-equals-example", _EXEC, "drift_converted", "fly from New York to Seattle on May 20",
         {"due": "2024-05-20"}, "due"),
    Case("pattern-equals-example", _EXEC, "drift_shelf", "put it on the first shelf", {"shelf": "#S1"}, "shelf"),
    Case("punctuation", _EXEC, "drift_code", "find the member Ann", {"name": "..."}, "name"),
    Case("unknown-marker-exec", _EXEC, "drift_code", "find twenty two books", {"name": "<UNKNOWN>"}, "name"),
    Case("escaped-unknown-marker", _EXEC, "drift_converted", "find twenty two books",
         {"due": "&lt;UNKNOWN&gt;"}, "due"),
    Case("number-word", _EXEC, "drift_rows", "show eight rows", {"count": 8}, "count"),
    Case("number-word-not-written", _EXEC, "drift_rows", "show eight rows", {"count": 9}, "count",
         difference="int-not-written"),
    Case("digits-inside-a-str", _EXEC, "drift_rows", "find Doyle and Sons 1",
         {"title": "Doyle and Sons 1", "count": 7}, "count"),
    # "One of:" in a description: one allowed-values rule on both paths (argument_checks.allowed_member, fix-48np.3)
    Case("one-of-outside", _EXEC, "drift_choice", "find the magazine Ann", {"kind_prose": "magazine"}, "kind_prose"),
    Case("one-of-spelling", _EXEC, "drift_choice", "find the member Ann", {"kind_prose": "Member"}, "kind_prose"),
    Case("one-of-member", _EXEC, "drift_choice", "find a branch near Ann", {"kind_prose": "branch"}, "kind_prose"),
]

_DIFF = "deliberate difference"

#: More cases for each deliberate difference.
DIFFERENCE_CASES = [
    # int-not-written
    Case("absent-int-one-number", _DIFF, "drift_rows", "list the loans of book b1 from row 10", {"count": None},
         "count", difference="int-not-written"),
    Case("int-not-written-default-written", _DIFF, "drift_page", "show 5 or 8 loans", {"size": 6}, "size",
         difference="int-not-written"),
    Case("int-agree-no-number", _DIFF, "drift_rows", "list the loans of the book", {"count": 20}, "count"),
    Case("int-agree-two-numbers", _DIFF, "drift_rows", "show 2 or 3 rows", {"count": 7}, "count"),
    # absent-int
    Case("absent-int-zero", _DIFF, "drift_rows", "show 2 or 3 rows", {"count": 0}, "count", difference="absent-int"),
    Case("absent-int-no-number", _DIFF, "drift_rows", "show the loans", {"count": None}, "count"),
    # exemptions
    Case("unoptioned-str-not-written", _DIFF, "book_details", "who has Foundation on loan", {"title": "Dune Messiah"},
         "title", difference="exemptions"),
    Case("unoptioned-str-invented", _DIFF, "transfer_loan", "Move Ann Lee's loan to the Main branch",
         {"member": "Ann Lee", "branch": "North"}, "branch", difference="exemptions"),
    Case("unoptioned-str-written", _DIFF, "book_details", "give me the details of Dune", {"title": "Dune"}, "title"),
    Case("unoptioned-str-placeholder", _DIFF, "book_details", "give me the details of it", {"title": "<title>"},
         "title"),
    # which-values
    Case("int-equal-to-default", _DIFF, "drift_page", "show 2 or 3 loans", {"size": 5}, "size",
         difference="which-values"),
    Case("int-equal-to-default-one-number", _DIFF, "drift_page", "show loans of 3 books", {"size": 5}, "size",
         difference="which-values"),
    # planning-only
    Case("subject-cut-short", _DIFF, "drift_subject", "Show the books of Ann Lee", {"name": "Ann"}, "name",
         difference="planning-only"),
    Case("subject-agree", _DIFF, "drift_subject", "Show the books of Ann Lee", {"name": "Ann Lee"}, "name"),
    Case("verify-verdict", _DIFF, "drift_verify", "is there a book named Lee", {"name": "Lee", "kind": "book"},
         "kind", verdicts={"kind": ("member", 0.9)}, difference="planning-only"),
    Case("verify-unsure-agree", _DIFF, "drift_verify", "is there a book named Lee", {"name": "Lee", "kind": "book"},
         "kind", verdicts={"kind": ("member", 0.3)}),
    Case("optional-repeats-another", _DIFF, "drift_code", "label the shelf of Ann Lee",
         {"name": "Ann Lee", "code": "Ann Lee"}, "code", difference="planning-only"),
    Case("identifier-whole-agree", _DIFF, "book_by_title", "who has acct_4421", {"title": "acct_4421"}, "title"),
]

CASES = CORPUS_CASES + EXECUTION_CASES + DIFFERENCE_CASES


# ---------------------------------------------------------------- the two checkers

@pytest.fixture(scope="module")
def commands(tmp_path_factory) -> dict[str, pc.PlanningCommandCommand]:
    """The tutorial's planning commands, with ``transfer_loan`` and the drift commands, loaded from a copy
    (the repo's ``___command_info`` is never read or written)."""
    copy = tmp_path_factory.mktemp("drift") / "tutorial"
    shutil.copytree(TUTORIAL, copy, ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    (copy / "_commands" / "transfer_loan.py").write_text(TRANSFER_LOAN, encoding="utf-8")
    for name, text in DRIFT_COMMANDS.items():
        (copy / "_commands" / f"{name}.py").write_text(text, encoding="utf-8")
    loaded = workflow_agent.load_planning_command_map(str(copy))
    return {name: dataclasses.replace(command, identifiers=ACCOUNTS) for name, command in loaded.items()}


@pytest.fixture(autouse=True)
def _framework(tmp_path):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    yield
    fastworkflow.RoutingRegistry.clear_registry()


def _extracted(model: type[BaseModel], values: dict) -> dict:
    """The extractor's answer for every field of *model*: the case's values, else its "not given" answer
    (NOT_FOUND for a required field, the default for an optional one), as the strictness test scripts it."""
    return {name: values.get(name, NOT_FOUND if info.is_required() else info.get_default(call_default_factory=True))
            for name, info in model.model_fields.items()}


def planning_verdict(command: pc.PlanningCommandCommand, case: Case):
    """The typed planning call's checks on the extractor's values (``extracted_arguments`` reads its "not
    given" text as no value; ``checked_arguments`` runs every check in order)."""
    raw = {name: pc._given_or_none(value) for name, value in _extracted(command.input_model, case.values).items()}
    param = next(p for p in command.params if p.name == case.field)
    try:
        args, notes = pc.checked_arguments(command, raw, case.request, None, case.verdicts)
    except pc.PlanningCommandCallError as refusal:
        text = str(refusal)
        if text == f"missing required argument {case.field}":
            return NOT_GIVEN
        if text.startswith(f"{case.field}="):
            return REFUSED
        return f"refused for another reason: {text}"
    given, value = raw[case.field], args[case.field]
    if pc._empty(given) or (given == param.default and value == param.default):
        return NOT_GIVEN if value == param.default else _as(value)
    if value == given:
        return ACCEPTED
    if any(note.startswith(f"{case.field}=") and note.endswith(": default") for note in notes):
        return REFUSED   # an optional argument a guard refused takes its default (its note says so)
    return _as(value)


def _literal_verdict(annotation, value):
    """The type check ``validate_parameters`` gives a ``Literal`` choice (its closure ``_try_coerce_literal``,
    restated): a member as written, else the one member it matches in any case."""
    members = get_args(annotation)
    if value in members:
        return ACCEPTED
    folded = [m for m in members if isinstance(m, str) and isinstance(value, str) and m.lower() == value.lower()]
    return _as(folded[0]) if len(folded) == 1 else REFUSED


def execution_verdict(command: pc.PlanningCommandCommand, case: Case):
    """The execution checker on the same ``Input``, as ``ParameterExtraction._llm_extract`` runs it: on the
    extractor's ``model_construct`` instance, against the command text; ``apply`` writes the outcome."""
    model = case.execution_model or command.input_model
    name = case.execution_field or case.field
    values = dict(case.values)
    if case.execution_field and case.field in values:
        values[case.execution_field] = values.pop(case.field)
    params = model.model_construct(**_extracted(model, values))
    info = model.model_fields[name]
    given = getattr(params, name)
    check = argument_checks.check_extracted_values(params, case.request)
    check.apply(params)
    if name in check.refused:
        return REFUSED
    if not argument_checks._given(info, given):
        return NOT_GIVEN
    if get_origin(info.annotation) is Literal:
        return _literal_verdict(info.annotation, given)
    value = getattr(params, name)
    return ACCEPTED if value == given else _as(value)


def _param(case: Case):
    marks = [pytest.mark.xfail(strict=True, reason=f"undecided drift: {case.drift}")] if case.drift else []
    return pytest.param(case, id=f"{case.source.split()[0]}:{case.id}", marks=marks)


@pytest.mark.parametrize("case", [_param(case) for case in CASES])
def test_the_checkers_agree_except_on_a_listed_difference(commands, case):
    command = commands[case.command]
    planning, execution = planning_verdict(command, case), execution_verdict(command, case)
    shown = (f"case {case.id!r} ({case.source}; {case.command}.{case.field}, request {case.request!r}, "
             f"values {case.values!r}): planning {_shown(planning)}, execution {_shown(execution)}")

    if case.difference is None:
        assert planning == execution, (
            f"The argument checkers disagree on {shown}. Align the checkers (one rule in "
            f"fastworkflow/utils/argument_checks.py, both paths calling it), or, by an explicit decision, add a "
            f"documented entry to DIFFERENCES in this file and tag the case with it (fix-48np.2's notes).")
        return
    difference = DIFFERENCES[case.difference]
    assert planning != execution and difference.shows(planning, execution), (
        f"{shown}; the case is listed as {case.difference!r} ({difference.what}), but does not show that "
        f"difference. Correct the entry or the case by an explicit decision; the list must stay truthful.")


def test_every_deliberate_difference_has_a_case():
    covered = {case.difference for case in CASES if case.difference}
    assert covered == set(DIFFERENCES), sorted(set(DIFFERENCES) - covered)


def test_every_strictness_corpus_case_is_in_the_table():
    """A case added to the strictness corpus is classified here (its field, and a difference or none)."""
    assert {case[0] for case in CORPUS} == set(CORPUS_FIELDS)


def test_the_table_holds_agreeing_refusals_and_acceptances(commands):
    """The table is not all differences: on many cases both checkers refuse, and on many both accept."""
    agreed = [planning_verdict(commands[c.command], c) for c in CASES if c.difference is None and c.drift is None]
    assert sum(v == REFUSED for v in agreed) >= 10
    assert sum(v == ACCEPTED or isinstance(v, tuple) for v in agreed) >= 10
