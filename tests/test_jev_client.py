"""Activation of the Jev-backed features (fix-4dsr, fix-xg1a): warnings, caching, key rotation,
the endpoint, the SDK's DEBUG logging, failure descriptions, and the SDK contract.

Real env vars and real ``TypeSafeClient`` construction. Calls go only to the
loopback stand-in (``jev_stub`` fixture, ``tests/jev_stub.py``) over real HTTP.

The contract test (``test_the_sdk_surface_fastworkflow_uses``) is the review
gate for bumping the exactly pinned typesafe-sdk: rerun it against the new
version before changing the pin.
"""
import json
import logging
import time
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.observability import capture_policy
from fastworkflow.observability import store as observability_store
from fastworkflow.observation_offloading import archive, finish_check, jev_client, search_router
from fastworkflow.observation_offloading.finish_check import CHECK_ENV, KEY_ENV, MODEL_ENV, checker_from_env
from fastworkflow.observation_offloading.search_router import ROUTER_ENV, router_for_workflow
from fastworkflow.turn_plan import PlanStep, PlanSubject, TurnPlan
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_agent import finish_check_active
from tests.jev_stub import choice_answer

try:
    import typesafe_sdk
    from typesafe_sdk import (
        Choice,
        Noul,
        TypeSafeAPIError,
        TypeSafeAPITimeoutError,
        TypeSafeAuthenticationError,
        TypeSafeBadRequestError,
        TypeSafeInternalServerError,
        TypeSafeRateLimitError,
        constants as sdk_constants,
    )
    from typesafe_sdk._core import logging as sdk_logging
except ImportError:  # the tests that need it skip
    typesafe_sdk = None

needs_sdk = pytest.mark.skipif(jev_client.TypeSafeClient is None, reason="typesafe-sdk not installed")
PINNED_SDK_VERSION = "0.7.2"


def _read_only(_command):
    """Every command declared read-only, so every step of the plan is checked."""
    return "read_only"


class _Collect(logging.Handler):
    def __init__(self, level=logging.WARNING):
        super().__init__(level=level)
        self.messages = []
        self.records = []

    def emit(self, record):
        self.records.append(record)
        self.messages.append(record.getMessage())


@pytest.fixture
def warnings_logged(monkeypatch):
    for name in (CHECK_ENV, ROUTER_ENV, KEY_ENV, MODEL_ENV, search_router.ROUTER_MODEL_ENV,
                 jev_client.BASE_URL_ENV, jev_client.SDK_BASE_URL_ENV,
                 observability_store.CAPTURE_PROFILE_VAR, archive.REDACTION_ENV):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delitem(fastworkflow._env_vars, name, raising=False)
    jev_client._WARNED.clear()
    finish_check._CHECKERS.clear()
    finish_check._MODEL_OVERRIDE_WARNED.clear()
    search_router._ROUTERS.clear()
    handler = _Collect()
    logger.addHandler(handler)
    yield handler.messages
    logger.removeHandler(handler)
    jev_client._WARNED.clear()
    finish_check._CHECKERS.clear()
    finish_check._MODEL_OVERRIDE_WARNED.clear()
    search_router._ROUTERS.clear()


def test_an_unset_flag_is_silent(warnings_logged, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "a-key-present-for-something-else")
    assert checker_from_env() is None
    assert router_for_workflow("wf") is None
    assert warnings_logged == []


@pytest.mark.parametrize("value", ["off", " OFF ", "0", "False", "no", "None", "", "  "])
def test_an_explicit_off_value_is_silent(warnings_logged, monkeypatch, value):
    monkeypatch.setenv(CHECK_ENV, value)
    monkeypatch.setenv(ROUTER_ENV, value)
    monkeypatch.setenv(KEY_ENV, "k1")
    assert checker_from_env() is None
    assert router_for_workflow("wf") is None
    assert warnings_logged == []


def test_an_unrecognised_value_warns_once_naming_the_fix(warnings_logged, monkeypatch):
    monkeypatch.setenv(CHECK_ENV, "on")
    monkeypatch.setenv(KEY_ENV, "k1")
    assert checker_from_env() is None
    assert checker_from_env() is None
    assert len(warnings_logged) == 1
    assert CHECK_ENV in warnings_logged[0] and "'on'" in warnings_logged[0]
    assert f"{CHECK_ENV}=jev" in warnings_logged[0]


@needs_sdk
def test_a_missing_key_warns_once_per_feature(warnings_logged, monkeypatch):
    monkeypatch.setenv(CHECK_ENV, "jev")
    monkeypatch.setenv(ROUTER_ENV, "jev")
    for _ in range(2):
        assert checker_from_env() is None
        assert router_for_workflow("wf") is None
    assert len(warnings_logged) == 2
    assert all(KEY_ENV in message for message in warnings_logged)
    assert CHECK_ENV in warnings_logged[0] and ROUTER_ENV in warnings_logged[1]


@pytest.mark.skipif(jev_client.TypeSafeClient is not None,
                    reason="typesafe-sdk is installed; the missing-SDK path needs an install without it")
def test_a_set_flag_without_the_sdk_warns_with_the_install_hint(warnings_logged, monkeypatch):
    monkeypatch.setenv(CHECK_ENV, "jev")
    monkeypatch.setenv(KEY_ENV, "k1")
    assert checker_from_env() is None
    assert len(warnings_logged) == 1
    assert "fastworkflow[jev]" in warnings_logged[0]


@needs_sdk
def test_a_rotated_key_builds_a_new_checker(warnings_logged, monkeypatch):
    monkeypatch.setenv(CHECK_ENV, "jev")
    monkeypatch.setenv(KEY_ENV, "first-key")
    first = checker_from_env()
    assert first is not None and checker_from_env() is first
    monkeypatch.setenv(KEY_ENV, "second-key")
    second = checker_from_env()
    assert second is not None and second is not first
    assert second._provider.client is not first._provider.client
    monkeypatch.setenv(MODEL_ENV, "jev-other")
    assert checker_from_env()._model == "jev-other"
    # The only warning is the one saying the overriding model is uncalibrated.
    assert len(warnings_logged) == 1 and MODEL_ENV in warnings_logged[0]


@needs_sdk
def test_a_rotated_key_or_model_builds_a_new_router(warnings_logged, monkeypatch):
    monkeypatch.setenv(ROUTER_ENV, "jev")
    monkeypatch.setenv(KEY_ENV, "first-key")
    first = router_for_workflow("wf")
    assert first is not None and router_for_workflow("wf") is first
    monkeypatch.setenv(KEY_ENV, "second-key")
    second = router_for_workflow("wf")
    assert second is not first
    monkeypatch.setenv(search_router.ROUTER_MODEL_ENV, "jev-other")
    assert router_for_workflow("wf")._model == "jev-other"


def test_the_cache_keeps_only_a_fingerprint_of_the_key():
    cache = jev_client.ClientCache()
    built = cache.get("model", "raw-secret-key", object)
    assert cache.get("model", "raw-secret-key", object) is built
    assert "raw-secret-key" not in repr(cache._entries)
    assert cache._entries["model"][0] == jev_client.key_fingerprint("raw-secret-key")


# ---------------------------------------------------------------------------
# The endpoint: FW_JEV_BASE_URL only
# ---------------------------------------------------------------------------

def _plan():
    return TurnPlan(steps=[PlanStep(text="Find Alan Cooper", commands=["find_identity"])],
                    subjects=[PlanSubject(name="Alan Cooper", kind="person")])


def _info_logged():
    handler = _Collect(level=logging.INFO)
    logger.addHandler(handler)
    return handler


@needs_sdk
def test_the_default_endpoint_is_typesafe_and_is_what_the_sdk_defaults_to(warnings_logged):
    assert jev_client.base_url() == jev_client.DEFAULT_BASE_URL == "https://api.typesafe.ai"
    assert jev_client.DEFAULT_BASE_URL == sdk_constants.DEFAULT_BASE_URL
    assert jev_client.SDK_BASE_URL_ENV == sdk_constants.BASE_URL_ENV
    assert warnings_logged == []


def test_an_https_endpoint_is_used_and_its_host_logged_once(warnings_logged, monkeypatch):
    monkeypatch.setenv(jev_client.BASE_URL_ENV, "https://jev.internal.example:8443/")
    info = _info_logged()
    try:
        assert jev_client.base_url() == "https://jev.internal.example:8443"
        assert jev_client.base_url() == "https://jev.internal.example:8443"
    finally:
        logger.removeHandler(info)
    hosts = [m for m in info.messages if "jev.internal.example:8443" in m]
    assert len(hosts) == 1 and jev_client.BASE_URL_ENV in hosts[0]
    assert warnings_logged == []


@needs_sdk
@pytest.mark.parametrize("url", ["http://evil.example", "http://evil.example:80/v1", "ftp://127.0.0.1",
                                 "https://user:hunter2@jev.example", "https://jev.example?token=hunter2",
                                 "https://", "https://jev.example:notaport"])
def test_an_endpoint_that_may_not_be_used_turns_every_feature_off_with_one_warning(
        warnings_logged, monkeypatch, url):
    monkeypatch.setenv(jev_client.BASE_URL_ENV, url)
    monkeypatch.setenv(CHECK_ENV, "jev")
    monkeypatch.setenv(ROUTER_ENV, "jev")
    monkeypatch.setenv(KEY_ENV, "k1")
    for _ in range(2):
        assert checker_from_env() is None
        assert router_for_workflow("wf") is None
    assert len(warnings_logged) == 1
    assert jev_client.BASE_URL_ENV in warnings_logged[0] and "stay off" in warnings_logged[0]
    assert "hunter2" not in warnings_logged[0]


@needs_sdk
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_plain_http_to_a_loopback_address_is_allowed(warnings_logged, monkeypatch, host):
    monkeypatch.setenv(jev_client.BASE_URL_ENV, f"http://{host}:8080")
    assert jev_client.base_url() == f"http://{host}:8080"
    assert warnings_logged == []


@needs_sdk
def test_the_features_reach_the_endpoint_fw_jev_base_url_names(warnings_logged, jev_stub, monkeypatch):
    monkeypatch.setenv(CHECK_ENV, "jev")
    monkeypatch.setenv(KEY_ENV, "stub-key")
    checker = checker_from_env()
    result = checker.check(_plan(), "Audit Alan Cooper", [], command_effect=_read_only)
    assert result.error is None and result.requests == len(jev_stub.requests) == 2
    assert jev_stub.requests[0]["headers"]["Authorization"] == "Bearer stub-key"
    assert warnings_logged == []


@needs_sdk
def test_typesafe_base_url_is_ignored_with_one_warning(warnings_logged, jev_stub, monkeypatch):
    # Nothing listens on port 9; were the SDK's variable used, the call would fail.
    monkeypatch.setenv(jev_client.SDK_BASE_URL_ENV, "http://127.0.0.1:9")
    monkeypatch.setenv(CHECK_ENV, "jev")
    monkeypatch.setenv(KEY_ENV, "stub-key")
    assert checker_from_env().check(_plan(), "Audit Alan Cooper", [], command_effect=_read_only).error is None
    assert checker_from_env() is not None
    assert len(jev_stub.requests) == 2
    assert len(warnings_logged) == 1
    assert jev_client.SDK_BASE_URL_ENV in warnings_logged[0] and jev_client.BASE_URL_ENV in warnings_logged[0]
    monkeypatch.delenv(jev_client.BASE_URL_ENV)
    assert jev_client.base_url() == jev_client.DEFAULT_BASE_URL


@needs_sdk
def test_a_changed_endpoint_builds_a_new_checker(warnings_logged, jev_stub, monkeypatch):
    monkeypatch.setenv(CHECK_ENV, "jev")
    monkeypatch.setenv(KEY_ENV, "stub-key")
    first = checker_from_env()
    monkeypatch.setenv(jev_client.BASE_URL_ENV, "https://jev.internal.example")
    assert checker_from_env() is not first


# ---------------------------------------------------------------------------
# The SDK's DEBUG logging
# ---------------------------------------------------------------------------

@needs_sdk
def test_the_sdks_debug_records_never_carry_the_payload(jev_stub, monkeypatch):
    marker = "payload-marker-alan.cooper@example.com"
    sdk_logger = logging.getLogger(jev_client.SDK_LOGGER)
    previous = sdk_logger.level
    captured = _Collect(level=logging.DEBUG)
    sdk_logger.addHandler(captured)
    # What TYPESAFE_LOG_LEVEL=debug does at SDK import.
    monkeypatch.setenv("TYPESAFE_LOG_LEVEL", "debug")
    sdk_logging.setup_logging()
    guard = next(f for f in sdk_logger.filters if isinstance(f, jev_client._DropWireBodies))
    try:
        assert sdk_logger.isEnabledFor(logging.DEBUG)
        client = jev_client.make_client("k1", "jev-test", 4.0, jev_stub.base_url)
        questions = {"q": Noul(instructions="Is it?", criteria={"true": "yes", "false": "no"})}
        client.system_one(state={"note": marker}, questions=questions)
        assert not any(marker in m for m in captured.messages)
        assert not any(r.levelno <= logging.DEBUG for r in captured.records)
        assert any("<- 200" in m and "req-1" in m for m in captured.messages)
        # Without the filter the same call would log the body: the test can fail.
        sdk_logger.removeFilter(guard)
        client.system_one(state={"note": marker}, questions=questions)
        assert any(marker in m for m in captured.messages)
    finally:
        if guard not in sdk_logger.filters:
            sdk_logger.addFilter(guard)
        sdk_logger.removeHandler(captured)
        sdk_logger.setLevel(previous)


# ---------------------------------------------------------------------------
# The capture policy: the features stay off where it would withhold what they send
# ---------------------------------------------------------------------------

def _turn_both_on(monkeypatch):
    monkeypatch.setenv(CHECK_ENV, "jev")
    monkeypatch.setenv(ROUTER_ENV, "jev")
    monkeypatch.setenv(KEY_ENV, "stub-key")


@needs_sdk
@pytest.mark.parametrize("env, value, cause", [
    (observability_store.CAPTURE_PROFILE_VAR, "evidence", "withholds command output"),
    (archive.REDACTION_ENV, "off", f"{archive.REDACTION_ENV}=off"),
])
def test_a_withholding_profile_or_redaction_off_keeps_both_features_off_with_one_warning_each(
        warnings_logged, jev_stub, monkeypatch, env, value, cause):
    _turn_both_on(monkeypatch)
    monkeypatch.setenv(env, value)
    for _ in range(2):
        checker = checker_from_env()
        assert checker is None
        assert router_for_workflow("wf") is None
    assert len(warnings_logged) == 2
    assert CHECK_ENV in warnings_logged[0] and ROUTER_ENV in warnings_logged[1]
    assert all(cause in message and "stays off" in message for message in warnings_logged)
    # With no checker the structured planner is not used either: the planner reverts to text.
    assert not finish_check_active(SimpleNamespace(finish_checker=checker))
    assert jev_stub.requests == []


@needs_sdk
def test_the_debug_profile_with_redaction_on_lets_both_features_on(warnings_logged, jev_stub, monkeypatch):
    _turn_both_on(monkeypatch)
    monkeypatch.setenv(observability_store.CAPTURE_PROFILE_VAR, "debug")
    monkeypatch.setenv(archive.REDACTION_ENV, "on")
    assert jev_client.withholding_cause() is None
    checker = checker_from_env()
    assert checker is not None and router_for_workflow("wf") is not None
    assert finish_check_active(SimpleNamespace(finish_checker=checker))
    assert warnings_logged == []


def test_an_unknown_capture_profile_is_a_withholding_cause(monkeypatch):
    monkeypatch.setenv(observability_store.CAPTURE_PROFILE_VAR, "evidnce")
    assert observability_store.CAPTURE_PROFILE_VAR in jev_client.withholding_cause()


def test_egress_scrubs_credentials_and_refuses_badges(monkeypatch):
    monkeypatch.delenv(observability_store.CAPTURE_PROFILE_VAR, raising=False)
    monkeypatch.delenv(archive.REDACTION_ENV, raising=False)
    token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    assert jev_client.egress(f"Audit {token}") == "Audit [REDACTED]"
    badge = json.dumps(capture_policy.evidence_policy().apply(
        observability_store.POLICY_PATH_OFFLOAD_OBSERVATION, "rows", classification="opaque-payload"))
    assert capture_policy.CAPTURE_ENVELOPE_MARKER in badge
    assert jev_client.egress(badge) is None
    assert jev_client.egress(f"Stored: {badge}") is None
    monkeypatch.setenv(observability_store.CAPTURE_PROFILE_VAR, "evidence")
    assert jev_client.egress("rows") is None
    monkeypatch.setenv(observability_store.CAPTURE_PROFILE_VAR, "debug")
    monkeypatch.setenv(archive.REDACTION_ENV, "off")
    assert jev_client.egress("rows") is None


@needs_sdk
def test_a_router_value_withheld_at_call_time_sends_nothing(warnings_logged, jev_stub, monkeypatch):
    _turn_both_on(monkeypatch)
    router = router_for_workflow("wf")
    assert router is not None
    badge = json.dumps(capture_policy.evidence_policy().apply(
        observability_store.POLICY_PATH_OFFLOAD_OBSERVATION, "rows", classification="opaque-payload"))
    route = router.route("list every row", "to report them", badge)
    assert (route["choice"], route["error"], route["error_stage"]) == (None, "policy_withheld", "redaction")
    assert not router.wants_all_rows(route)
    # The profile changed after the router was built: the backstop still holds.
    monkeypatch.setenv(observability_store.CAPTURE_PROFILE_VAR, "evidence")
    assert router.route("list every row", "to report them", "1 row\nuid  label")["error"] == "policy_withheld"
    assert jev_stub.requests == []
    assert warnings_logged == []


# ---------------------------------------------------------------------------
# Describing and warning about failures
# ---------------------------------------------------------------------------

def _failure(error_type="TypeSafeRateLimitError", status=429):
    return {"error_type": error_type, "status": status, "request_id": "req-9", "code": "rate_limit_exceeded"}


def _warnings_from(warner, failures, pause=0.0):
    handler = _Collect()
    logger.addHandler(handler)
    try:
        for failure in failures:
            warner.warn(failure)
            if pause:
                time.sleep(pause)
    finally:
        logger.removeHandler(handler)
    return handler.messages


def test_a_warner_logs_each_kind_of_failure_once_per_interval():
    warner = jev_client.FailureWarner("the feature", "it is skipped", interval_seconds=3600)
    messages = _warnings_from(warner, [_failure(), _failure(), _failure(), _failure(status=500),
                                       _failure("TypeSafeAPITimeoutError", None), _failure(status=500)])
    assert len(messages) == 3
    assert messages[0] == ("the feature unavailable (TypeSafeRateLimitError, status 429, request req-9, "
                           "code rate_limit_exceeded); it is skipped")


def test_a_warner_with_no_interval_logs_every_failure():
    warner = jev_client.FailureWarner("the feature", "it is skipped", interval_seconds=0)
    assert len(_warnings_from(warner, [_failure()] * 3)) == 3


def test_a_sustained_failure_is_logged_again_with_how_many_went_unlogged():
    warner = jev_client.FailureWarner("the feature", "it is skipped", interval_seconds=0.2)
    first = _warnings_from(warner, [_failure()] * 3)
    time.sleep(0.25)
    later = _warnings_from(warner, [_failure()])
    assert len(first) == 1 and len(later) == 1
    assert "(2 more like it since the last warning)" in later[0]


def test_an_error_code_is_a_token_never_free_text():
    assert jev_client.error_code({"error": {"type": "max_tokens_exceeded", "message": "Too long."}}) \
        == "max_tokens_exceeded"
    assert jev_client.error_code({"detail": {"code": "invalid_key"}}) == "invalid_key"
    assert jev_client.error_code({"error": "rate_limited"}) == "rate_limited"
    assert jev_client.error_code({"error": {"message": "Unknown subject Alan Cooper"}}) is None
    assert jev_client.error_code({"error": "Something about Alan Cooper went wrong"}) is None
    assert jev_client.error_code({"detail": "Internal error"}) is None
    assert jev_client.error_code("plain text body") is None
    assert jev_client.describe_failure(ValueError("secret text")) == {
        "error_type": "ValueError", "status": None, "request_id": None, "code": None}


# ---------------------------------------------------------------------------
# The SDK contract (the review gate for a typesafe-sdk bump)
# ---------------------------------------------------------------------------

def _sdk_error(client, jev_stub, status, payload):
    jev_stub.respond = lambda _body: (status, payload)
    questions = {"q": Noul(instructions="Is it?", criteria={"true": "yes", "false": "no"})}
    with pytest.raises(TypeSafeAPIError) as caught:
        client.system_one(state={"k": "v"}, questions=questions)
    return caught.value


@needs_sdk
def test_the_sdk_surface_fastworkflow_uses(jev_stub):
    assert typesafe_sdk.__version__ == PINNED_SDK_VERSION

    # Construction: api_key, model, timeout, retry (one attempt), base_url.
    client = jev_client.make_client("contract-key", "jev-contract", 2.0, jev_stub.base_url)
    # The SDK has no public timeout attribute; ``_configured_timeout`` reads the
    # private ``_config.timeout``, and without it silently falls back.
    assert client._config.timeout == 2.0
    assert jev_client._configured_timeout(client) == 2.0
    noul = Noul(instructions="Is it?", criteria={"true": "yes", "false": "no"})
    choice = Choice(instructions="Which?", criteria={"a": "the first", "b": "the second"})
    jev_stub.answer = lambda _name, q: 0.25 if q["type"] == "noul" else choice_answer("b", q["criteria"], 0.8)
    response = client.system_one(state={"k": "v"}, questions={"n": noul, "c": choice})
    assert response.answers["n"].noul == 0.25
    assert response.answers["c"].choice == "b"
    assert response.answers["c"].probabilities["b"] == 0.8
    assert (response.usage.input_tokens, response.usage.output_tokens) == (7, 1)
    assert response.usage.model_dump() == {"input_tokens": 7, "output_tokens": 1}

    # The wire request: path, bearer key, body shape.
    sent = jev_stub.requests[0]
    assert sent["path"] == "/v1/systemone"
    assert sent["headers"]["Authorization"] == "Bearer contract-key"
    assert sent["body"] == {
        "state": {"k": "v"}, "model": "jev-contract",
        "questions": {"n": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes", "false": "no"}},
                      "c": {"type": "choice", "instructions": "Which?",
                            "criteria": {"a": "the first", "b": "the second"}}}}

    # Error classes: status, body, request_id; str() hides a code behind a human message.
    too_long = {"error": {"type": "max_tokens_exceeded", "message": "The request is longer than the model accepts."}}
    error = _sdk_error(client, jev_stub, 400, too_long)
    assert isinstance(error, TypeSafeBadRequestError)
    assert (error.status, error.body, error.request_id) == (400, too_long, "req-2")
    assert "max_tokens_exceeded" not in str(error)
    assert jev_client.state_too_large(error)
    assert jev_client.describe_failure(error) == {"error_type": "TypeSafeBadRequestError", "status": 400,
                                                  "request_id": "req-2", "code": "max_tokens_exceeded"}
    for status, error_class in ((401, TypeSafeAuthenticationError), (429, TypeSafeRateLimitError),
                                (500, TypeSafeInternalServerError)):
        before = len(jev_stub.requests)
        error = _sdk_error(client, jev_stub, status, {"error": {"type": "e", "message": "m"}})
        assert type(error) is error_class and error.status == status
        assert not jev_client.state_too_large(error)
        assert len(jev_stub.requests) == before + 1, "one attempt, no retries"

    # A timeout is the SDK's own type, a TimeoutError, with no HTTP details.
    jev_stub.respond = None
    jev_stub.delay = 1.0
    slow = jev_client.make_client("contract-key", "jev-contract", 0.3, jev_stub.base_url)
    with pytest.raises(TypeSafeAPITimeoutError) as caught:
        slow.system_one(state={"k": "v"}, questions={"n": noul})
    assert isinstance(caught.value, TimeoutError)
    assert jev_client.describe_failure(caught.value)["status"] is None
    assert "contract-key" not in json.dumps(jev_client.describe_failure(caught.value))
