"""Production wiring tests for workflow-specific synthetic-generation personas.

These tests enter through ``train_workflow`` and use temporary copies of a golden
workflow. No framework component is mocked and no shipped workflow is modified.
"""

import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest

from fastworkflow.command_context_model import (
    CommandContextModel,
    CommandContextModelValidationError,
)
from fastworkflow.train.__main__ import (
    _with_workflow_persona_source,
    train_workflow,
)
from fastworkflow.train.personas import (
    DomainConditionedPersonaSource,
    PersonaConfigError,
    get_persona_source,
)


HELLO_WORLD_PATH = Path("tests/hello_world_workflow")


def _copy_hello_world(tmp_path: Path) -> Path:
    workflow_path = tmp_path / "hello_world"
    shutil.copytree(
        HELLO_WORLD_PATH,
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info",
            "___workflow_contexts",
            "___convo_info",
            "__pycache__",
        ),
    )
    return workflow_path


def _write_personas(workflow_path: Path, payload: dict) -> None:
    (workflow_path / "personas.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_train_workflow_reads_the_workflow_persona_file(tmp_path):
    workflow_path = _copy_hello_world(tmp_path)
    _write_personas(
        workflow_path,
        {"schema_version": 999, "personas": ["A calculator user."]},
    )

    with pytest.raises(PersonaConfigError, match="schema_version"):
        train_workflow(str(workflow_path))

    assert get_persona_source() is None


def test_train_workflow_tears_down_personas_after_orchestration_failure(tmp_path):
    workflow_path = _copy_hello_world(tmp_path)
    _write_personas(
        workflow_path,
        {"domain_keywords": ["calculator", "arithmetic"]},
    )
    (workflow_path / "_commands/context_inheritance_model.json").write_text(
        "{not valid json",
        encoding="utf-8",
    )

    observed_sources = []

    def observe_context_model_load(frame, event, _arg):
        if (
            event == "call"
            and frame.f_code is CommandContextModel.load.__func__.__code__
        ):
            observed_sources.append(get_persona_source())

    sys.setprofile(observe_context_model_load)
    try:
        with pytest.raises(CommandContextModelValidationError, match="Invalid JSON"):
            train_workflow(str(workflow_path))
    finally:
        sys.setprofile(None)

    assert observed_sources
    assert isinstance(observed_sources[0], DomainConditionedPersonaSource)
    assert get_persona_source() is None


def test_train_workflow_constructs_one_source_and_uses_finally():
    wiring_source = inspect.getsource(_with_workflow_persona_source)

    assert train_workflow.__wrapped__ is not train_workflow
    assert wiring_source.count(
        "personas.persona_source_for_workflow(workflow_path)"
    ) == 1
    assert wiring_source.count("personas.set_persona_source(") == 2
    assert "finally:" in wiring_source
