"""One switch turns the model response cache off for every LM the framework builds.

FW_LM_CACHE=0 (the experiment harness sets it) must reach the Jev classifier LM as well as the LMs
``dspy_utils.get_lm`` builds; otherwise repeated attempts of a task replay an earlier attempt's Jev
answers and run-to-run variance is understated.
"""

import pytest

import fastworkflow
from fastworkflow import workflow_agent
from fastworkflow.utils import dspy_utils


@pytest.fixture
def env(monkeypatch):
    saved = dict(fastworkflow._env_vars)
    workflow_agent._jev_lm.cache_clear()
    yield fastworkflow._env_vars
    fastworkflow._env_vars.clear()
    fastworkflow._env_vars.update(saved)
    workflow_agent._jev_lm.cache_clear()


@pytest.mark.parametrize("value, enabled", [("0", False), ("false", False), ("1", True), (None, True)])
def test_the_switch_reads_the_env(env, monkeypatch, value, enabled):
    env.pop("FW_LM_CACHE", None)
    monkeypatch.delenv("FW_LM_CACHE", raising=False)
    if value is not None:
        env["FW_LM_CACHE"] = value
    assert dspy_utils.lm_cache_enabled() is enabled


@pytest.mark.parametrize("value, cached", [("0", False), ("1", True)])
def test_the_jev_lm_follows_the_switch(env, value, cached):
    env["FW_LM_CACHE"] = value
    env["JEV_API_KEY"] = "test-key-not-used"
    lm = workflow_agent._jev_lm()
    assert lm is not None
    assert getattr(lm, "cache") is cached


def test_get_lm_follows_the_switch(env):
    env["FW_LM_CACHE"] = "0"
    env["LLM_TEST_MODEL"] = "openai/gpt-test"
    lm = dspy_utils.get_lm("LLM_TEST_MODEL", "LITELLM_API_KEY_TEST")
    assert getattr(lm, "cache") is False
