"""Chatbot picker Train button (fix-kw7.16): detached spawn + poll.

Stub subprocesses only — never ``train_main``, never bundled
``fastworkflow/examples/*/___command_info``.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fastworkflow.run_chatbot import launcher
from fastworkflow.run_chatbot import server as run_chatbot_server


STUB_SLEEP = """\
import time
time.sleep({sleep})
"""

STUB_WRITE_THRESHOLD = """\
import json, os, time
time.sleep({sleep})
info = os.path.join({wf!r}, "___command_info")
os.makedirs(os.path.join(info, "global"), exist_ok=True)
with open(os.path.join(info, "routing_definition.json"), "w", encoding="utf-8") as f:
    json.dump({{"contexts": {{"*": ["demo"]}}}}, f)
with open(os.path.join(info, "global", "threshold.json"), "w", encoding="utf-8") as f:
    json.dump({{"confidence_threshold": 0.5}}, f)
"""


def _write_stub(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def _env_pair(wf: Path) -> tuple[str, str]:
    env = wf / "fastworkflow.env"
    passwords = wf / "fastworkflow.passwords.env"
    env.write_text("LLM_SYNDATA_GEN=mistral/mistral-small-latest\n", encoding="utf-8")
    passwords.write_text("LITELLM_API_KEY_SYNDATA_GEN=test-key\n", encoding="utf-8")
    return str(env), str(passwords)


def _untrained_workflow(root: Path, name: str = "app_wf") -> Path:
    wf = root / name
    (wf / "_commands").mkdir(parents=True)
    return wf


def _mark_trained(wf: Path) -> None:
    info = wf / "___command_info"
    (info / "global").mkdir(parents=True)
    (info / "routing_definition.json").write_text(
        json.dumps({"contexts": {"*": ["demo"], "IntentDetection": ["x"]}}),
        encoding="utf-8",
    )
    (info / "global" / "threshold.json").write_text(
        json.dumps({"confidence_threshold": 0.5}), encoding="utf-8"
    )


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    for _ in range(20):
        if not launcher.process_is_alive(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


@pytest.fixture
def state_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "state"
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(root))
    return root


# ----------------------------------------------------------------------
# Filesystem trained-check (no torch)
# ----------------------------------------------------------------------


class TestWorkflowIsTrained:
    def test_command_info_dir_alone_is_not_trained(self, tmp_path):
        wf = _untrained_workflow(tmp_path)
        (wf / "___command_info").mkdir()
        assert run_chatbot_server._workflow_is_trained(str(wf)) is False

    def test_threshold_json_for_global_is_trained(self, tmp_path):
        wf = _untrained_workflow(tmp_path)
        _mark_trained(wf)
        assert run_chatbot_server._workflow_is_trained(str(wf)) is True

    def test_cme_contexts_in_routing_are_ignored(self, tmp_path):
        wf = _untrained_workflow(tmp_path)
        _mark_trained(wf)
        # IntentDetection is listed in routing (as in hello_world) but is a CME
        # context — missing IntentDetection/threshold.json must not un-train it.
        assert not (wf / "___command_info" / "IntentDetection" / "threshold.json").exists()
        assert run_chatbot_server._workflow_is_trained(str(wf)) is True

    def test_local_untrained_candidate_is_trainable(self, tmp_path, state_root, monkeypatch):
        wf = _untrained_workflow(tmp_path, "local_app")
        monkeypatch.chdir(tmp_path)
        mine = next(
            w
            for w in run_chatbot_server.list_workflow_candidates()
            if w["name"] == "local_app"
        )
        assert mine["path"] == str(wf)
        assert mine["source"] == "local"
        assert mine["trained"] is False
        assert mine["training"] is False
        assert mine["trainable"] is True


# ----------------------------------------------------------------------
# plan_train_spawn refusals (no subprocess)
# ----------------------------------------------------------------------


class TestPlanTrainSpawn:
    def test_refuses_bundled_package_examples(self, tmp_path, state_root):
        bundled = tmp_path / "pkg_examples" / "hello_world"
        bundled.mkdir(parents=True)
        (bundled / "_commands").mkdir()
        plan = launcher.plan_train_spawn(
            workflow_path=str(bundled),
            env_file_path=str(tmp_path / "e"),
            passwords_file_path=str(tmp_path / "p"),
            bundled_root=str(tmp_path / "pkg_examples"),
            live_train=None,
            datasets_missing=False,
        )
        assert plan.ok is False
        assert "bundled" in plan.reason.lower()

    def test_refuses_missing_env_files(self, tmp_path, state_root):
        wf = _untrained_workflow(tmp_path)
        plan = launcher.plan_train_spawn(
            workflow_path=str(wf),
            env_file_path="",
            passwords_file_path="",
            bundled_root=str(tmp_path / "not_examples"),
            live_train=None,
            datasets_missing=False,
        )
        assert plan.ok is False
        assert "Select this workflow" in plan.reason

    def test_refuses_when_another_train_is_live(self, tmp_path, state_root):
        wf = _untrained_workflow(tmp_path)
        env, passwords = _env_pair(wf)
        pid_path, _log = launcher.train_artifact_paths(str(wf), create=True)
        plan = launcher.plan_train_spawn(
            workflow_path=str(wf),
            env_file_path=env,
            passwords_file_path=passwords,
            bundled_root=str(tmp_path / "not_examples"),
            live_train=(pid_path, 99999),
            datasets_missing=False,
        )
        assert plan.ok is False
        assert "already running" in plan.reason.lower()

    def test_refuses_missing_datasets(self, tmp_path, state_root):
        wf = _untrained_workflow(tmp_path)
        env, passwords = _env_pair(wf)
        plan = launcher.plan_train_spawn(
            workflow_path=str(wf),
            env_file_path=env,
            passwords_file_path=passwords,
            bundled_root=str(tmp_path / "not_examples"),
            live_train=None,
            datasets_missing=True,
        )
        assert plan.ok is False
        assert "datasets" in plan.reason

    def test_ok_builds_train_module_command(self, tmp_path, state_root):
        wf = _untrained_workflow(tmp_path)
        env, passwords = _env_pair(wf)
        plan = launcher.plan_train_spawn(
            workflow_path=str(wf),
            env_file_path=env,
            passwords_file_path=passwords,
            bundled_root=str(tmp_path / "not_examples"),
            live_train=None,
            datasets_missing=False,
        )
        assert plan.ok is True
        assert plan.cmd[:3] == [sys.executable, "-m", "fastworkflow.train"]
        assert plan.cmd[3:] == [str(wf.resolve()), env, passwords]


# ----------------------------------------------------------------------
# Real stub subprocess
# ----------------------------------------------------------------------


class TestSpawnDetachedTrain:
    def test_writes_pid_redirects_log_and_survives_chatbot_shutdown(
        self, tmp_path, state_root
    ):
        wf = _untrained_workflow(tmp_path)
        stub = _write_stub(tmp_path / "stub.py", STUB_SLEEP.format(sleep=20))
        pid_path, log_path = launcher.train_artifact_paths(str(wf), create=True)
        plan = launcher.TrainSpawnPlan(
            ok=True,
            cmd=[sys.executable, str(stub)],
            workflow_path=str(wf),
            pid_path=pid_path,
            log_path=log_path,
        )
        pid = launcher.spawn_detached_train(plan)
        try:
            assert pid > 0
            assert launcher.read_pid_file(pid_path) == pid
            assert launcher.process_is_alive(pid)
            assert launcher.is_train_running(str(wf)) is True
            assert log_path.endswith(launcher.TRAIN_LOG_FILENAME)

            srv = run_chatbot_server.ChatbotServer(
                port=0, spawn_options={"no_server": True}
            )
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            srv.shutdown()
            thread.join(timeout=5)
            assert launcher.process_is_alive(pid)
        finally:
            _kill(pid)

    def test_second_live_train_is_refused(self, tmp_path, state_root):
        wf = _untrained_workflow(tmp_path)
        stub = _write_stub(tmp_path / "stub.py", STUB_SLEEP.format(sleep=20))
        pid_path, log_path = launcher.train_artifact_paths(str(wf), create=True)
        plan = launcher.TrainSpawnPlan(
            ok=True,
            cmd=[sys.executable, str(stub)],
            workflow_path=str(wf),
            pid_path=pid_path,
            log_path=log_path,
        )
        pid = launcher.spawn_detached_train(plan)
        try:
            env, passwords = _env_pair(wf)
            refused = launcher.plan_train_spawn(
                workflow_path=str(wf),
                env_file_path=env,
                passwords_file_path=passwords,
                bundled_root=str(tmp_path / "not_examples"),
                datasets_missing=False,
            )
            assert refused.ok is False
            assert "already running" in refused.reason.lower()
        finally:
            _kill(pid)

    def test_poll_flips_to_trained_when_stub_writes_threshold(
        self, tmp_path, state_root
    ):
        wf = _untrained_workflow(tmp_path)
        stub = _write_stub(
            tmp_path / "stub.py",
            STUB_WRITE_THRESHOLD.format(sleep=0.2, wf=str(wf)),
        )
        pid_path, log_path = launcher.train_artifact_paths(str(wf), create=True)
        plan = launcher.TrainSpawnPlan(
            ok=True,
            cmd=[sys.executable, str(stub)],
            workflow_path=str(wf),
            pid_path=pid_path,
            log_path=log_path,
        )
        pid = launcher.spawn_detached_train(plan)
        try:
            deadline = time.time() + 5
            while time.time() < deadline:
                if (
                    not launcher.is_train_running(str(wf))
                    and run_chatbot_server._workflow_is_trained(str(wf))
                ):
                    break
                time.sleep(0.05)
            assert run_chatbot_server._workflow_is_trained(str(wf)) is True
            assert launcher.is_train_running(str(wf)) is False
        finally:
            _kill(pid)


# ----------------------------------------------------------------------
# HTTP control plane
# ----------------------------------------------------------------------


def _post(server, path, body, token=...):
    if token is ...:
        token = server.token
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


class TestTrainHttp:
    @pytest.fixture
    def server(self, tmp_path, state_root):
        srv = run_chatbot_server.ChatbotServer(port=0, spawn_options={"no_server": True})
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        yield srv
        srv.shutdown()
        thread.join(timeout=5)

    def test_train_requires_token(self, server, tmp_path):
        wf = _untrained_workflow(tmp_path)
        status, body = _post(server, "/api/train", {"path": str(wf)}, token=None)
        assert status == 401
        assert "token" in body["error"]

    def test_train_refuses_bundled_examples(self, server):
        import fastworkflow

        bundled = str(
            Path(fastworkflow.__file__).resolve().parent / "examples" / "hello_world"
        )
        status, body = _post(server, "/api/train", {"path": bundled})
        assert status == 400
        assert "bundled" in body["error"].lower()
        # Must not have started a process against the packaged artifacts.
        assert launcher.is_train_running(bundled) is False

    def test_train_refuses_missing_env_and_points_at_picker(self, server, tmp_path):
        wf = _untrained_workflow(tmp_path)
        status, body = _post(server, "/api/train", {"path": str(wf)})
        assert status == 400
        assert "Select this workflow" in body["error"]

    def test_post_train_spawns_stub_and_lists_training(
        self, server, tmp_path, monkeypatch
    ):
        wf = _untrained_workflow(tmp_path)
        _env_pair(wf)
        stub = _write_stub(tmp_path / "stub.py", STUB_SLEEP.format(sleep=20))
        real_plan = launcher.plan_train_spawn

        def plan_with_stub(**kwargs):
            kwargs.setdefault("datasets_missing", False)
            kwargs.setdefault("bundled_root", str(tmp_path / "not_examples"))
            plan = real_plan(**kwargs)
            if plan.ok:
                plan.cmd = [sys.executable, str(stub)]
            return plan

        monkeypatch.setattr(launcher, "plan_train_spawn", plan_with_stub)
        status, body = _post(server, "/api/train", {"path": str(wf)})
        pid = body.get("pid")
        try:
            assert status == 200, body
            assert body["training"] is True
            assert pid and launcher.process_is_alive(pid)
            assert launcher.is_train_running(str(wf)) is True

            status2, body2 = _post(server, "/api/train", {"path": str(wf)})
            assert status2 == 409
            assert "already running" in body2["error"].lower()
        finally:
            if pid:
                _kill(pid)


class TestPidReuseHardening:
    def test_recycled_pid_is_not_mistaken_for_a_train(self, tmp_path, monkeypatch):
        """A live pid with a DIFFERENT kernel start time is a stranger wearing
        a recycled number: it must not report training, and the stale pid file
        must be cleared so it can never block training globally."""
        import os

        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        wf = tmp_path / "app_wf"
        wf.mkdir()
        pid_path, _log = launcher.train_artifact_paths(str(wf), create=True)
        # Our own (live) pid, but recorded with a fabricated start time.
        with open(pid_path, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} 1")
        assert launcher.is_train_running(str(wf)) is False
        assert not os.path.exists(pid_path)

    def test_finished_train_pid_file_is_cleared_on_next_check(
        self, tmp_path, monkeypatch
    ):
        import os
        import subprocess

        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
        wf = tmp_path / "app_wf"
        wf.mkdir()
        pid_path, _log = launcher.train_artifact_paths(str(wf), create=True)
        proc = subprocess.Popen(["true"])
        launcher.write_pid_file(pid_path, proc.pid)
        proc.wait()
        assert launcher.is_train_running(str(wf)) is False
        assert not os.path.exists(pid_path)
