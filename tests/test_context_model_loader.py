import json
import os
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.context_model_loader import ContextModelLoader, ContextModelLoaderError


@pytest.fixture(scope="session", autouse=True)
def _init_fastworkflow():
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})


def _write_tmp_model(tmp_path: Path, data: dict) -> Path:
    file_path = tmp_path / "_commands/context_inheritance_model.json"
    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data))
    return file_path


def test_valid_model_load(tmp_path):
    model_data = {
        "inheritance": {"*": {"base": []}},
        "aggregation": {"A": {"container": ["*"]}},
    }
    model_path = _write_tmp_model(tmp_path, model_data)
    loader = ContextModelLoader(model_path)

    loaded = loader.load()
    assert loaded == model_data
    assert loader.inheritance == model_data["inheritance"]
    assert loader.aggregation == model_data["aggregation"]


def test_missing_aggregation_is_added(tmp_path):
    model_data = {"inheritance": {"*": {"base": []}}}
    model_path = _write_tmp_model(tmp_path, model_data)

    loader = ContextModelLoader(model_path)
    loaded = loader.load()

    assert "aggregation" in loaded and loaded["aggregation"] == {}


def test_missing_inheritance_raises(tmp_path):
    model_data = {"aggregation": {}}
    model_path = _write_tmp_model(tmp_path, model_data)

    loader = ContextModelLoader(model_path)
    with pytest.raises(ContextModelLoaderError, match="inheritance"):
        loader.load()


def test_invalid_json(tmp_path):
    bad_file = tmp_path / "_commands/context_inheritance_model.json"
    # Create parent directories if they don't exist
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("{ invalid json }")
    loader = ContextModelLoader(bad_file)
    with pytest.raises(ContextModelLoaderError):
        loader.load() 