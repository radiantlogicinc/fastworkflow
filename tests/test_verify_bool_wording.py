"""The `verify` check's question and options for a bool (fix-jd9o.9).

A bool that defaults to true is asked in the value form ("Which value should the argument take?"):
the earlier "Does the request ask against this?" double negative was read near coin-flip (0.47 on
one field, 0.70 across default-true bools, against 0.97 for the value form). A bool that defaults
to false keeps its wording: neither form won there (each failed a different case).
"""
from types import SimpleNamespace

from fastworkflow import planning_commands


def _bool(default):
    return SimpleNamespace(type="bool", default=default, name="include_details",
                           desc="include the detail rows", values=(), means=())


def test_a_default_true_bool_is_offered_in_the_value_form():
    options = dict(planning_commands._check_options(_bool(True)))
    assert options == {
        "true": "true: the request does not mention include_details, or says or implies it",
        "false": "false: the request says not to, or says the opposite of include_details",
    }


def test_a_default_false_bool_keeps_its_wording():
    options = dict(planning_commands._check_options(_bool(False)))
    assert options == {
        "true": "yes: include the detail rows",
        "false": "no: the request does not ask for this, or says not to",
    }


def test_the_default_true_question_asks_which_value_the_argument_takes(monkeypatch):
    seen = {}

    class _Predict:
        def __init__(self, signature):
            seen["question"] = signature.output_fields["value"].json_schema_extra["desc"]

        def __call__(self, **_):
            choice = SimpleNamespace(value="true", probabilities={"true": 0.9}, confidence=0.9)
            return SimpleNamespace(value=choice)

    import dspy
    from dspy.utils import DummyLM

    monkeypatch.setattr(dspy, "Predict", _Predict)
    command = SimpleNamespace(description="Generate a report.")
    value, probability = planning_commands._check_one(DummyLM([]), command, _bool(True), "a summary report")
    assert seen["question"] == ("Which value should the argument 'include_details' take? "
                                "It means: include the detail rows.")
    assert (value, probability) == (True, 0.9)
