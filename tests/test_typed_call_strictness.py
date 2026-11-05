"""The typed planning call is no looser with the shared extractor than it was with its own fill (fix-48np, gate 2a).

``typed_planning_call`` now takes its arguments from the shared parameter extractor (``extracted_arguments``)
and keeps the planning checks it had (``check_arguments``, ``_validate_arguments``). Over one corpus of
planning-command calls, gathered from the planning tests' argument cases plus adversarial ones (a
placeholder, an invented subject, a value the request does not write, an identifier fragment, a role
conflict, the extractor's own "not given" text), every value the planning checks refuse is refused by the
call, on the first answer and on the retry, and every value they accept gives the same arguments.

Integration: a copy of ``tests/planning_tutorial_workflow`` (with the two-role ``transfer_loan`` command of
the argument-screen tests added), its real planning commands, the real DSPy extractor and the real checks.
The model boundary is two scripted ``DummyLM``\\s (the extractor's and the planner's). No test calls a model.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import uuid
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

import fastworkflow
from fastworkflow import planning_commands as pc
from fastworkflow import workflow_agent
from fastworkflow.utils import signatures
from fastworkflow.utils.signatures import NOT_FOUND
from tests.test_planning_command_argument_screens import TRANSFER_LOAN

TUTORIAL = Path(__file__).resolve().parent / "planning_tutorial_workflow"
ACCOUNTS = (re.compile(r"\bacct_\d+\b"),)

#: (id, planning command, request, the extractor's values). The values are what the extractor answers, on the
#: first call and on the retry alike, so a refused value stays refused.
CORPUS = [
    # accepted values (the checks pass them, perhaps respelled)
    ("title", "book_by_title", "who wrote Dune", {"title": "Dune"}),
    ("title-case", "book_loans", "who has Dune Messiah on loan", {"title": "dune messiah"}),
    ("person", "author_books", "What do you have by Frank Herbert?", {"author": "Frank Herbert"}),
    ("two-roles", "transfer_loan", "Move Ann Lee's loan to the Main branch", {"member": "Ann Lee", "branch": "Main"}),
    ("pronoun-one-subject", "author_books",
     "What do you have by Frank Herbert, and how many copies does it have?", {"author": "it"}),
    ("enum", "find_record", "is there a member named Lee", {"name": "Lee", "kind": "member"}),
    ("bools", "book_report", "summarise the loans and shelf for Dune", {"title": "Dune", "loans": True, "brief": False}),
    ("whole-identifier", "book_by_title", "who has acct_4421", {"title": "4421"}),
    ("optional-placeholder", "transfer_loan", "Move Ann Lee's loan to the Main branch",
     {"member": "Ann Lee", "branch": "Main", "note": "<entity_name>"}),
    # placeholders (test_planning_command_argument_screens)
    ("braces", "author_books", "Which books did the author write?", {"author": "{AUTHOR_PLACEHOLDER}"}),
    ("angle", "author_books", "Which books did the author write?", {"author": "<entity_name>"}),
    ("name-number", "author_books", "Which books did the author write?", {"author": "NAME_1"}),
    ("unspecified", "author_books", "Which books did the author write?", {"author": "unspecified_right"}),
    ("invented-example", "author_books", "Which books did the author write?", {"author": "John Doe"}),
    # pronouns with no single subject
    ("that-one", "author_books", "Which books did the author of that one write?", {"author": "that one"}),
    ("it", "author_books", "Which books did the author of that one write?", {"author": "it"}),
    # vocabulary words
    ("own-name", "author_books", "Which books did the author write?", {"author": "author"}),
    ("field-name", "find_books", "Which titles by Asimov?", {"query": "title"}),
    ("command-word", "transfer_loan", "Move the loan to the Main branch", {"member": "loan", "branch": "Main"}),
    ("declared-choice", "transfer_loan", "Move Ann Lee's loan to the account branch",
     {"member": "Ann Lee", "branch": "account"}),
    ("kind-word", "find_books", "Which book by Asimov?", {"query": "book"}),
    # adversarial
    ("invented-subject", "author_books", "list the books by Ursula Le Guin", {"author": "Stephen King"}),
    ("not-in-request", "book_loans", "who has Foundation on loan", {"title": "Dune Messiah"}),
    ("deictic", "book_by_title", "who wrote this book", {"title": "this book"}),
    ("identifier-fragment", "book_by_title", "who has acct_4421 or acct_4422", {"title": "acct_44"}),
    ("role-conflict", "transfer_loan", "Move the loan of member Ann Lee to branch Ann Lee",
     {"member": "Ann Lee", "branch": "Ann Lee"}),
    ("role-conflict-unwritten", "transfer_loan", "Ann Lee has a loan at Ann Lee",
     {"member": "Ann Lee", "branch": "Ann Lee"}),
    ("not-an-enum-value", "find_record", "is there a member named Lee", {"name": "Lee", "kind": "author"}),
    ("not-given-text", "book_details", "give me the details of this book", {"title": NOT_FOUND}),
    ("unknown-marker", "book_details", "give me the details of this book", {"title": "<UNKNOWN>"}),
    ("missing-required", "author_books", "show every book by this author", {"author": ""}),
]


@pytest.fixture(scope="module")
def folder(tmp_path_factory) -> str:
    copy = tmp_path_factory.mktemp("strictness") / "tutorial"
    shutil.copytree(TUTORIAL, copy, ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    (copy / "_commands" / "transfer_loan.py").write_text(TRANSFER_LOAN, encoding="utf-8")
    return str(copy)


@pytest.fixture(autouse=True)
def _framework(tmp_path, monkeypatch):
    fastworkflow.init({"FASTWORKFLOW_STATE_ROOT": str(tmp_path / "workflow_contexts")})
    fastworkflow.RoutingRegistry.clear_registry()
    monkeypatch.setitem(fastworkflow._env_vars, "FW_JEV_FIELD_GIVEN", "off")
    monkeypatch.setattr(signatures, "LLM_PARAM_EXTRACTION", "scripted/extractor")
    yield
    fastworkflow.RoutingRegistry.clear_registry()


class _CountingLM(DummyLM):
    """Answers in order and counts the requests (in a list: a copy of the LM shares it)."""

    def __init__(self, answers):
        super().__init__(answers, adapter=dspy.JSONAdapter())
        self.requests: list = []

    @property
    def calls(self) -> int:
        return len(self.requests)

    def forward(self, prompt=None, messages=None, **kwargs):
        self.requests.append(messages or prompt)
        return super().forward(prompt=prompt, messages=messages, **kwargs)


def _command(folder: str, name: str) -> pc.PlanningCommandCommand:
    command = workflow_agent.load_planning_command_map(folder)[name]
    return dataclasses.replace(command, identifiers=ACCOUNTS)


def _before(command: pc.PlanningCommandCommand, values: dict, request: str):
    """The planning checks alone, on the values as the planner gave them before (the extractor's "not given"
    text is no value there either: the planner left such a field empty)."""
    raw = {p.name: values.get(p.name) for p in command.params}
    raw = {k: (None if v in (NOT_FOUND, "<UNKNOWN>") else v) for k, v in raw.items()}
    try:
        args, notes = pc.check_arguments(command, raw, request, {})
        return ("ok", pc._validate_arguments(command, args, request, None, notes))
    except (pc.PlanningCommandInputError, pc.PlanningCommandCallError) as refused:
        return ("refused", str(refused))


def _after(command: pc.PlanningCommandCommand, values: dict, request: str, folder: str, monkeypatch,
           typed: dict | None = None):
    """The new call on *values* from the extractor (first answer and retry). The planner's typed fill, which the
    fallback reads for a required argument still missing or refused, gives *typed*: by default the same values,
    the worst case (the fallback finds nothing better than the refused value)."""
    answer = {"reasoning": "scripted", **{p.name: values.get(p.name, NOT_FOUND if p.required else p.default)
                                         for p in command.params}}
    extractor = _CountingLM([answer, answer])
    monkeypatch.setattr(signatures.dspy_utils, "get_lm", lambda *a, **k: extractor)
    fill = {p.name: (values if typed is None else typed).get(p.name, "" if p.required else p.default)
            for p in command.params if not p.alone}
    # one answer serves either planner call: the typed fill (when the fallback fires) and the step proposal
    step = {"reasoning": "r", **fill, "steps_before": [], "steps_after": []}
    planner = DummyLM([step, step])
    wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"strict_{uuid.uuid4().hex[:8]}")
    try:
        with dspy.context(lm=planner):
            call = pc.typed_planning_call(command, request, "", workflow=wf)
    except (pc.PlanningCommandInputError, pc.PlanningCommandCallError) as refused:
        return ("refused", str(refused)), extractor
    return ("ok", call.args), extractor


@pytest.mark.parametrize("name, request_text, values", [case[1:] for case in CORPUS], ids=[c[0] for c in CORPUS])
def test_the_typed_call_refuses_what_the_planning_checks_refuse(folder, monkeypatch, name, request_text, values):
    command = _command(folder, name)
    before = _before(command, values, request_text)
    after, extractor = _after(command, values, request_text, folder, monkeypatch)

    if before[0] == "refused":
        assert after[0] == "refused", (before, after)
        # refused, then read once more and refused again (a value its type cannot hold is refused at parse, retried)
        assert extractor.calls >= 2, after
    else:
        assert after == before


def test_the_corpus_holds_refusals_of_every_kind(folder):
    """The corpus is not all accepted values: each adversarial kind is refused by the planning checks."""
    refused = {case[0] for case in CORPUS if _before(_command(folder, case[1]), case[3], case[2])[0] == "refused"}
    assert {"braces", "angle", "name-number", "unspecified", "invented-example", "that-one", "it", "own-name",
            "field-name", "command-word", "declared-choice", "kind-word", "invented-subject", "not-in-request",
            "identifier-fragment", "role-conflict", "role-conflict-unwritten", "not-an-enum-value",
            "not-given-text", "unknown-marker", "missing-required"} <= refused


# ---------------------------------------------------------------- the typed-fill fallback (fix-48np)


def test_the_fallback_fills_a_required_argument_the_extractor_missed(folder, monkeypatch):
    command = _command(folder, "author_books")
    request = "list every Austen book we own with its copies"
    after, extractor = _after(command, {"author": NOT_FOUND}, request, folder, monkeypatch, typed={"author": "Austen"})

    assert after == ("ok", {"author": "Austen"})
    assert extractor.calls >= 2                     # the extractor was asked, and asked again, first


def test_the_fallback_cannot_add_a_value_the_checks_refuse(folder, monkeypatch):
    command = _command(folder, "author_books")
    after, _ = _after(command, {"author": NOT_FOUND}, "list the books by Ursula Le Guin", folder, monkeypatch,
                      typed={"author": "Stephen King"})

    assert after[0] == "refused" and "Stephen King" in after[1]


@pytest.mark.parametrize("typed", [NOT_FOUND, "<UNKNOWN>", ""])
def test_the_fallback_reads_the_typed_fills_not_given_text_as_no_value(folder, monkeypatch, typed):
    command = _command(folder, "book_details")          # title carries no in_request: only emptiness refuses it
    after, _ = _after(command, {"title": NOT_FOUND}, "give me the details of this book", folder, monkeypatch,
                      typed={"title": typed})

    assert after == ("refused", "missing required argument title")


def test_the_fallback_never_overrides_a_value_the_extractor_gave(folder, monkeypatch):
    command = _command(folder, "transfer_loan")
    request = "Move Ann Lee's loan to the Main branch"
    after, _ = _after(command, {"member": "Ann Lee", "branch": NOT_FOUND}, request, folder, monkeypatch,
                      typed={"member": "Main", "branch": "Main"})

    assert after[0] == "ok"
    assert after[1]["member"] == "Ann Lee" and after[1]["branch"] == "Main"
