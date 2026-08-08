"""Integration tests for the live-session cap and checkpoint retention (Release C).

`MAX_LIVE_SESSIONS` is an operator control, and an operator control whose
documented default outranks the container override is not one. `get_env_var`
returns a supplied default *before* consulting `os.environ`, so a resolver that
passed a default would make the container variable unreachable — the value would
silently be whatever the workflow env file said.

Lowering the cap is also what makes eviction routine, which is what makes
checkpoint retention necessary: a unique-channel workload writes a snapshot per
request that is never read again.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import uuid

import pytest

import fastworkflow
from fastworkflow.run_fastapi_mcp import checkpoint
from fastworkflow.run_fastapi_mcp.utils import (
    DEFAULT_MAX_LIVE_SESSIONS,
    ChannelSessionManager,
    resolve_max_live_sessions,
)


@pytest.fixture
def hello_world_workflow_path():
    package_path = fastworkflow.get_fastworkflow_package_path()
    workflow_path = os.path.join(package_path, "examples", "hello_world")
    if not os.path.isdir(workflow_path):
        pytest.skip(f"hello_world workflow not found at {workflow_path}")
    return workflow_path


@pytest.fixture
def env_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI tests")
    return env_file, passwords_file


@pytest.fixture
def app_module(hello_world_workflow_path, env_files):
    env_file, passwords_file = env_files
    sys.argv = [
        "pytest",
        "--workflow_path",
        hello_world_workflow_path,
        "--env_file_path",
        env_file,
        "--passwords_file_path",
        passwords_file,
    ]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from dotenv import dotenv_values

    fastworkflow.init({**dotenv_values(env_file), **dotenv_values(passwords_file)})
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    return main


@pytest.fixture(autouse=True)
def clean_process_env(monkeypatch):
    monkeypatch.delenv("MAX_LIVE_SESSIONS", raising=False)


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_the_default_is_the_lowered_cap():
    """Release C's whole point: 50, not 2000."""
    value, source = resolve_max_live_sessions()

    assert value == DEFAULT_MAX_LIVE_SESSIONS == 50
    assert source == "default"


def test_the_process_environment_wins(monkeypatch):
    """The container variable must be reachable, which is the reason for the resolver.

    get_env_var returns a supplied default before it ever looks at os.environ, so
    a resolver written the obvious way would ignore this.
    """
    monkeypatch.setenv("MAX_LIVE_SESSIONS", "137")

    value, source = resolve_max_live_sessions()

    assert value == 137
    assert source == "process environment"


def test_the_resolved_source_is_reported_not_just_the_value(monkeypatch):
    """An operator has to be able to see whether their override actually took effect."""
    monkeypatch.setenv("MAX_LIVE_SESSIONS", "12")
    assert resolve_max_live_sessions()[1] == "process environment"

    monkeypatch.delenv("MAX_LIVE_SESSIONS")
    assert resolve_max_live_sessions()[1] == "default"


@pytest.mark.parametrize("bad", ["0", "-5", "abc", "3.5"])
def test_an_unusable_value_fails_startup_rather_than_traffic(monkeypatch, bad):
    """A cap that is going to reject requests should reject the process instead."""
    monkeypatch.setenv("MAX_LIVE_SESSIONS", bad)

    with pytest.raises(ValueError, match="MAX_LIVE_SESSIONS"):
        resolve_max_live_sessions()


def test_an_empty_value_falls_through_rather_than_crashing(monkeypatch):
    """An unset-looking variable is a common deployment artifact, not a config error."""
    monkeypatch.setenv("MAX_LIVE_SESSIONS", "")

    value, source = resolve_max_live_sessions()

    assert value == DEFAULT_MAX_LIVE_SESSIONS
    assert source == "default"


def test_the_manager_applies_the_resolver_and_reports_the_source(monkeypatch):
    """Resolution happens after init(), so the manager cannot bake it in at import."""
    manager = ChannelSessionManager()
    monkeypatch.setenv("MAX_LIVE_SESSIONS", "3")

    value, source = manager.configure_max_live_sessions()

    assert (value, source) == (3, "process environment")
    assert manager.max_live_sessions == 3
    assert manager.max_live_sessions_source == "process environment"


def test_the_env_template_documents_the_knob_without_shadowing_it():
    """An active assignment here would outrank the container variable.

    That is the exact defect the resolver exists to prevent, reintroduced through
    the file the resolver reads second.
    """
    package_path = fastworkflow.get_fastworkflow_package_path()
    template = os.path.join(package_path, "examples", "fastworkflow.env")
    lines = [
        line.strip()
        for line in open(template, encoding="utf-8").read().splitlines()
        if "MAX_LIVE_SESSIONS" in line
    ]

    assert lines, "the knob is undocumented"
    assert all(line.startswith("#") for line in lines), (
        "MAX_LIVE_SESSIONS is assigned in the workflow env template, which would "
        "shadow the deployment's own value"
    )


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def test_retention_never_reclaims_a_channel_the_process_still_holds(app_module):
    """The store cannot see live runtimes; the server has to name them.

    Reclaiming a live channel's checkpoint would turn its next eviction into the
    silent state loss this whole release exists to remove.
    """
    channel_id = _channel("protected")

    async def body():
        manager = app_module.session_manager
        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        # Give it a checkpoint on disk, so there is something to reclaim.
        assert manager.checkpoint_for_shutdown([]) >= 1

        # Retain nothing by age or count, so only the protection can spare it.
        from fastworkflow.checkpoint_store import RetentionPolicy

        report = manager.reap_checkpoints(
            RetentionPolicy(max_age_seconds=1e-6, max_channels=0)
        )
        # Assert about THIS channel rather than the namespace: the store is
        # shared with every other test in the run, and their abandoned channels
        # are exactly what a reaper is supposed to reclaim.
        survived = manager.checkpoint_store.load_for_adoption(
            deployment_id=checkpoint.deployment_id(),
            workflow_fingerprint=checkpoint.workflow_fingerprint(
                app_module.ARGS.workflow_path
            ),
            channel_id=channel_id,
        )
        return report, channel_id in manager._sessions, survived

    report, still_live, survived = asyncio.run(body())

    assert still_live
    assert report.protected_channels >= 1, "the live channel was not protected"
    assert survived is not None, (
        "retention reclaimed the checkpoint of a channel the process still holds live"
    )


def test_the_namespace_can_be_measured_in_bytes(app_module):
    """Section 16.5 gates Release C on total bytes reaching a plateau.

    A rate is not a bound: any positive rate grows forever, so the soak harness
    needs a real byte figure walked from the files themselves.
    """
    async def body():
        manager = app_module.session_manager
        stats = manager.checkpoint_store.stats()
        return stats

    stats = asyncio.run(body())

    assert hasattr(stats, "total_bytes_physical")
    assert hasattr(stats, "total_bytes_apparent")
    assert stats.total_bytes_apparent >= 0
    assert stats.describe()
