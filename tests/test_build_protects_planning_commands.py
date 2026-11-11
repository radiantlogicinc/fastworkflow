"""Build generator: the generated-file header, planning-command refusals, the planning-aware
validator, and the name-collision skip. Real files in tmp_path; no mocks, no LLM or network calls."""
import logging
from pathlib import Path

import pytest

from fastworkflow.build.class_analysis_structures import ClassInfo, FunctionInfo, MethodInfo
from fastworkflow.build.command_file_generator import (
    generate_command_files,
    validate_command_file_components_in_dir,
    validate_command_imports,
)
from fastworkflow.build.command_file_template import (
    GENERATED_FILE_HEADER,
    create_command_file,
    create_function_command_file,
)

PLANNING_SOURCE = '''class Signature:
    """Say where the book is held."""

    plain_utterances = ["where is this book kept"]


class PlanGenerator:
    def shelf(self):
        return []
'''

EXECUTION_SOURCE_WITHOUT_HEADER = '''class Signature:
    class Input(BaseModel):
        pass

    class Output(BaseModel):
        success: bool

    plain_utterances = ["run widget"]


class ResponseGenerator:
    def __call__(self, workflow, command, input):
        return None  # old generated body
'''


@pytest.fixture
def warnings_log(caplog):
    """Capture fastWorkflow warnings: that logger does not propagate to root, so attach the handler directly."""
    log = logging.getLogger("fastWorkflow")
    log.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="fastWorkflow"):
            yield caplog
    finally:
        log.removeHandler(caplog.handler)


def _class_and_method(class_name: str, method_name: str) -> tuple[ClassInfo, MethodInfo]:
    class_info = ClassInfo(class_name, "application/module.py")
    method_info = MethodInfo(method_name, [{"name": "widget_id", "annotation": "int"}], docstring="Run it.")
    class_info.methods.append(method_info)
    return class_info, method_info


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_overwrite_true_leaves_planning_command_byte_identical_and_warns(tmp_path, warnings_log):
    commands = tmp_path / "_commands"
    planning = _write(commands / "Shelf" / "run.py", PLANNING_SOURCE)
    before = planning.read_bytes()
    class_info, _ = _class_and_method("Shelf", "Run")  # would write Shelf/run.py

    generate_command_files({"Shelf": class_info}, str(commands), str(tmp_path), overwrite=True)

    assert planning.read_bytes() == before
    assert "it is a planning command (defines PlanGenerator)" in warnings_log.text
    assert str(planning) in warnings_log.text


def test_overwrite_true_refuses_planning_function_command(tmp_path, warnings_log):
    commands = tmp_path / "_commands"
    planning = _write(commands / "shelf_report.py", PLANNING_SOURCE)
    before = planning.read_bytes()
    function_info = FunctionInfo("shelf_report", "application/module.py", [{"name": "x", "annotation": "int"}])

    create_function_command_file(function_info, str(commands), file_name="shelf_report.py", source_dir=str(tmp_path), overwrite=True)

    assert planning.read_bytes() == before
    assert "it is a planning command (defines PlanGenerator)" in warnings_log.text


def test_new_generated_files_start_with_the_header(tmp_path):
    commands = tmp_path / "_commands"
    (commands / "Widget").mkdir(parents=True)
    class_info, method_info = _class_and_method("Widget", "Fetch")
    class_path = create_command_file(class_info, method_info, str(commands / "Widget"), file_name="fetch.py", source_dir=str(tmp_path))
    function_info = FunctionInfo("global_fn", "application/module.py", [{"name": "x", "annotation": "int"}])
    fn_path = create_function_command_file(function_info, str(commands), file_name="global_fn.py", source_dir=str(tmp_path))

    for path in (class_path, fn_path):
        assert Path(path).read_text(encoding="utf-8").splitlines()[0] == GENERATED_FILE_HEADER


def test_header_less_execution_file_is_still_overwritten(tmp_path):
    commands = tmp_path / "_commands"
    old = _write(commands / "Widget" / "fetch.py", EXECUTION_SOURCE_WITHOUT_HEADER)
    class_info, method_info = _class_and_method("Widget", "Fetch")

    create_command_file(class_info, method_info, str(commands / "Widget"), file_name="fetch.py", source_dir=str(tmp_path), overwrite=True)

    text = old.read_text(encoding="utf-8")
    assert text.splitlines()[0] == GENERATED_FILE_HEADER
    assert "# old generated body" not in text


def test_validator_accepts_planning_command_with_only_signature_and_plan_generator(tmp_path):
    commands = tmp_path / "_commands"
    _write(commands / "Shelf" / "run.py", PLANNING_SOURCE)
    # Control: an execution-shaped file with no ResponseGenerator still gets the execution checks
    _write(commands / "Shelf" / "fetch.py", EXECUTION_SOURCE_WITHOUT_HEADER.replace("class ResponseGenerator", "class Other"))

    errors = validate_command_file_components_in_dir(str(commands))

    flagged = {Path(path).name for path, _ in errors}
    assert "run.py" not in flagged
    assert "fetch.py" in flagged


def test_validator_still_requires_signature_on_a_planning_command(tmp_path):
    commands = tmp_path / "_commands"
    _write(commands / "Shelf" / "run.py", "class PlanGenerator:\n    def shelf(self):\n        return []\n")

    errors = validate_command_file_components_in_dir(str(commands))

    assert [msg for path, msg in errors if path.endswith("run.py")] == ["Missing Signature class"]


def test_planning_command_is_imported_as_a_command(tmp_path):
    commands = tmp_path / "_commands"
    _write(commands / "Shelf" / "run.py", PLANNING_SOURCE)

    assert validate_command_imports(str(commands)) is True


def test_name_collision_with_planning_command_is_skipped_and_warns(tmp_path, warnings_log):
    commands = tmp_path / "_commands"
    # Qualified name Shelf/run is the planning command; the generator would write the same path
    planning = _write(commands / "Shelf" / "run.py", PLANNING_SOURCE)
    before = planning.read_bytes()
    class_info, method_info = _class_and_method("Shelf", "Run")

    create_command_file(class_info, method_info, str(commands / "Shelf"), file_name="run.py", source_dir=str(tmp_path), overwrite=True)

    assert planning.read_bytes() == before
    assert "it is a planning command (defines PlanGenerator)" in warnings_log.text
    assert str(planning) in warnings_log.text


def test_unparseable_planning_command_is_not_overwritten(tmp_path, warnings_log):
    commands = tmp_path / "_commands"
    # Half-edited: a syntax error, but a top-level PlanGenerator is still textually present
    half_edited = "class Signature:\n    plain_utterances = [\n\nclass PlanGenerator:\n    def shelf(self) -> \n"
    planning = _write(commands / "Shelf" / "run.py", half_edited)
    before = planning.read_bytes()
    class_info, method_info = _class_and_method("Shelf", "Run")

    create_command_file(class_info, method_info, str(commands / "Shelf"), file_name="run.py", source_dir=str(tmp_path), overwrite=True)

    assert planning.read_bytes() == before
    assert "it is a planning command (defines PlanGenerator)" in warnings_log.text
