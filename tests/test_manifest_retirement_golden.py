"""Golden comparison for the manifest-retirement epic (fix-d3eh).

Pins what the IDO workflow tells an agent and a planner about its contexts:

* the enterable-context list and the planner's all-contexts display text
  (``command_metadata_api``), and
* the unavailable-command route message for a fixed set of (context, command)
  pairs (``context_navigation.unavailable_command_message``).

The fixture was captured with ``workflow_runtime.json`` present. Later beads
must keep these outputs identical. The IDO workflow is copied to ``tmp_path``
(without its caches and secrets) so the test never writes into the IDO tree.

Regenerate the fixture only deliberately, with::

    uv run --no-sync python tests/test_manifest_retirement_golden.py --write
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

IDO_WORKFLOW = Path(__file__).resolve().parents[2] / "ido" / "ido_workflow"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "manifest_retirement_golden.json"


@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch):
    """get_module() prepends the workflow's parent dir (~/rl/ido) to sys.path; undo it per test."""
    monkeypatch.setattr(sys, "path", list(sys.path))


class _Workflow:
    """The attributes the navigation helpers read, with no live context objects."""

    def __init__(self, folderpath: str, current: str) -> None:
        self.folderpath = folderpath
        self.current_command_context_name = current
        self.current_command_context = None
        self.root_command_context = None
        self.is_current_command_context_root = True

    def get_parent(self, obj):
        return None


def _copy_workflow(destination: Path) -> str:
    shutil.copytree(
        IDO_WORKFLOW,
        destination,
        ignore=shutil.ignore_patterns(
            "___command_info", "___convo_info", "*.passwords.env", "fastworkflow.env*"
        ),
    )
    return str(destination)


def _compute(workflow_path: str, pairs: list[dict]) -> dict:
    import fastworkflow
    from fastworkflow.command_metadata_api import CommandMetadataAPI
    from fastworkflow.context_navigation import unavailable_command_message

    fastworkflow.init({"NOT_FOUND": "NOT_FOUND"})
    occupiable = CommandMetadataAPI._occupiable_context_names(workflow_path)
    display = CommandMetadataAPI.get_all_contexts_command_display_text(
        subject_workflow_path=workflow_path,
        cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
        active_context_name="*",
        navigation_workflow=_Workflow(workflow_path, "*"),
    )
    messages = [
        {
            "context": pair["context"],
            "token": pair["token"],
            "homes": pair["homes"],
            "message": unavailable_command_message(
                _Workflow(workflow_path, pair["context"]),
                pair["token"],
                pair["context"],
                pair["homes"],
            ),
        }
        for pair in pairs
    ]
    return {
        "enterable_contexts": None if occupiable is None else sorted(occupiable),
        "enterable_display_text": display,
        "unavailable_messages": messages,
    }


def _write_fixture(pairs: list[dict]) -> None:
    """Capture the golden outputs from the IDO workflow as it stands now."""
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        workflow = _copy_workflow(Path(scratch) / "ido_workflow")
        captured = _compute(workflow, pairs)
    golden = {
        "source": "ido/ido_workflow with workflow_runtime.json present (fix-d3eh.1)",
        **captured,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.skipif(
    not IDO_WORKFLOW.is_dir(),
    reason="ido workflow is not checked out beside fastworkflow",
)
def test_ido_enterable_list_and_unavailable_messages_match_golden(tmp_path):
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pairs = [
        {"context": m["context"], "token": m["token"], "homes": m["homes"]}
        for m in golden["unavailable_messages"]
    ]
    workflow = _copy_workflow(tmp_path / "ido_workflow")

    actual = _compute(workflow, pairs)

    assert actual["enterable_contexts"] == golden["enterable_contexts"]
    assert actual["enterable_display_text"] == golden["enterable_display_text"]
    assert actual["unavailable_messages"] == golden["unavailable_messages"]


if __name__ == "__main__":
    # Re-captures the outputs for the pairs already recorded in the fixture. The
    # pair selection is fixed in the fixture; editing it is a deliberate change.
    if sys.argv[1:] != ["--write"] or not FIXTURE.is_file():
        sys.exit("usage: test_manifest_retirement_golden.py --write (fixture must exist)")
    _recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))["unavailable_messages"]
    _write_fixture(
        [{"context": m["context"], "token": m["token"], "homes": m["homes"]} for m in _recorded]
    )
    print(f"wrote {FIXTURE}")
