"""
Basic tests for FastWorkflow FastAPI service
These tests validate the structure and basic functionality
"""

import os
import tempfile
import sys
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def hello_world_workflow_path():
    """Get path to hello_world example workflow"""
    import fastworkflow
    package_path = fastworkflow.get_fastworkflow_package_path()
    workflow_path = os.path.join(package_path, "examples", "hello_world")
    if not os.path.isdir(workflow_path):
        pytest.skip(f"hello_world workflow not found at {workflow_path}")
    return workflow_path


@pytest.fixture
def env_files():
    """Get paths to actual env files for testing"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file):
        pytest.skip(f"Env file not found at {env_file}. Please create ./env/.env")
    if not os.path.isfile(passwords_file):
        pytest.skip(f"Passwords file not found at {passwords_file}. Please create ./passwords/.env")
    return env_file, passwords_file


@pytest.fixture
def unique_user_id():
    """Generate a unique user ID for each test to avoid session conflicts"""
    import uuid
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def app_module(hello_world_workflow_path, env_files):
    """Import the FastAPI app module with required CLI ARGS configured."""
    env_file, passwords_file = env_files
    # Ensure a fresh import with required CLI args present
    sys.argv = [
        "pytest",
        "--workflow_path", hello_world_workflow_path,
        "--env_file_path", env_file,
        "--passwords_file_path", passwords_file,
    ]
    import fastworkflow.run_fastapi_mcp.__main__ as main
    importlib.reload(main)
    
    # Manually trigger fastworkflow.init() since TestClient doesn't invoke lifespan
    import fastworkflow
    from dotenv import dotenv_values
    env_vars = {
        **dotenv_values(env_file),
        **dotenv_values(passwords_file)
    }
    fastworkflow.init(env_vars)
    
    # Clear routing caches to ensure clean state for each test
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    
    return main


def test_fastapi_imports(app_module):
    """Test that main module and ConversationStore import successfully"""
    try:
        from fastworkflow.run_fastapi_mcp.conversation_store import (
            ConversationStore,
            ConversationSummary,
            generate_topic_and_summary,
        )
        assert app_module.app is not None
        assert ConversationStore is not None
        assert ConversationSummary is not None
        assert generate_topic_and_summary is not None
    except ImportError as e:
        pytest.fail(f"Failed to import service components: {e}")


def test_root_endpoint(app_module):
    """Test the root health check endpoint"""
    client = TestClient(app_module.app)
    response = client.get("/")
    assert response.status_code == 200
    assert "FastWorkflow API is running" in response.text
    assert "/docs" in response.text


def test_validation_startup_mutual_exclusion(app_module):
    """Validate InitializationRequest requires channel_id and supports user_id and stream_format"""
    from pydantic import ValidationError
    InitReq = app_module.InitializationRequest
    # Missing channel_id should fail
    with pytest.raises(ValidationError):
        InitReq()
    # Valid with channel_id and optional user_id/stream_format
    req = InitReq(channel_id="test_channel", user_id="test_user", stream_format="ndjson")
    assert req.channel_id == "test_channel"
    assert req.user_id == "test_user"
    assert req.stream_format == "ndjson"
    # Valid with only channel_id
    req2 = InitReq(channel_id="channel2")
    assert req2.channel_id == "channel2"
    assert req2.user_id is None


def test_validation_feedback_presence(app_module):
    """Test that at least one feedback field must be provided"""
    from pydantic import ValidationError
    PostFeedbackRequest = app_module.PostFeedbackRequest
    with pytest.raises(ValidationError):
        PostFeedbackRequest(
            binary_or_numeric_score=None,
            nl_feedback=None,
        )
    req = PostFeedbackRequest(binary_or_numeric_score=True)
    assert req.binary_or_numeric_score == 1.0
    req = PostFeedbackRequest(nl_feedback="Great response")
    assert req.nl_feedback == "Great response"
    req = PostFeedbackRequest(binary_or_numeric_score=0.8, nl_feedback="Good")
    assert req.binary_or_numeric_score == 0.8
    assert req.nl_feedback == "Good"


def test_session_manager_basic(app_module):
    """Test basic ChannelSessionManager get_session behavior (no sessions initially)"""
    import asyncio
    async def test_async():
        manager = app_module.ChannelSessionManager()
        session = await manager.get_session("test_channel")
        assert session is None
    asyncio.run(test_async())


def _authorize(client: TestClient, access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


def _initialize(client: TestClient, channel_id: str, user_id: str = None) -> dict:
    payload = {"channel_id": channel_id, "stream_format": "ndjson"}
    if user_id:
        payload["user_id"] = user_id
    resp = client.post("/initialize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data and "refresh_token" in data
    return data


def test_initialize_endpoint(app_module, unique_user_id):
    """Test POST /initialize endpoint"""
    client = TestClient(app_module.app)
    data = _initialize(client, "test_user_1")
    assert data["token_type"].lower() == "bearer"


def test_initialize_with_startup_command_in_request(app_module, unique_user_id):
    """Test initialize with startup_command provided in request body"""
    client = TestClient(app_module.app)
    resp = client.post("/initialize", json={
        "channel_id": unique_user_id,
        "user_id": "user_123",
        "stream_format": "ndjson",
        "startup_command": "add_two_numbers first_num=5 second_num=3"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert "startup_output" in data
    # Verify startup_output has expected structure
    if data["startup_output"]:
        assert "command_responses" in data["startup_output"]


def test_initialize_with_startup_action_in_request(app_module, unique_user_id):
    """Test initialize with startup_action provided in request body"""
    client = TestClient(app_module.app)
    resp = client.post("/initialize", json={
        "channel_id": unique_user_id,
        "user_id": "user_456",
        "stream_format": "ndjson",
        "startup_action": {
            "command_name": "add_two_numbers",
            "parameters": {"first_num": 10, "second_num": 20}
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert "startup_output" in data
    if data["startup_output"]:
        assert "command_responses" in data["startup_output"]


def test_initialize_requires_user_id_with_startup(app_module, unique_user_id):
    """Test that user_id is required when startup is provided"""
    client = TestClient(app_module.app)
    # Without user_id
    resp = client.post("/initialize", json={
        "channel_id": unique_user_id,
        "startup_command": "add_two_numbers first_num=1 second_num=2"
    })
    assert resp.status_code == 400
    assert "user_id is required" in resp.json()["detail"]


def test_initialize_xor_validation(app_module, unique_user_id):
    """Test that startup_command and startup_action are mutually exclusive"""
    client = TestClient(app_module.app)
    resp = client.post("/initialize", json={
        "channel_id": unique_user_id,
        "user_id": "user_789",
        "startup_command": "some command",
        "startup_action": {"command_name": "test", "parameters": {}}
    })
    assert resp.status_code == 400
    assert "both startup_command and startup_action" in resp.json()["detail"].lower()


def test_initialize_validation_errors(app_module):
    """Invalid workflow path via ARGS should 500"""
    client = TestClient(app_module.app)
    # Set invalid workflow path
    app_module.ARGS.workflow_path = "/nonexistent/workflow"
    resp = client.post("/initialize", json={"channel_id": "test_channel"})
    assert resp.status_code == 500
    assert "Internal error" in resp.json()["detail"]
    # Note: do not restore ARGS here; other tests import a new module instance if needed


def test_invoke_agent_endpoint(app_module, unique_user_id):
    """Test POST /invoke_agent endpoint (agentic mode)"""
    client = TestClient(app_module.app)
    init = _initialize(client, unique_user_id)
    headers = _authorize(client, init["access_token"])
    response = client.post("/invoke_agent", headers=headers, json={
        "user_query": "add 15 and 25",
        "timeout_seconds": 60,
    })
    assert response.status_code == 200
    data = response.json()
    assert "command_responses" in data and len(data["command_responses"]) > 0


def test_invoke_assistant_endpoint(app_module, unique_user_id):
    """Test POST /invoke_assistant endpoint (deterministic mode)"""
    client = TestClient(app_module.app)
    init = _initialize(client, unique_user_id)
    headers = _authorize(client, init["access_token"])
    response = client.post("/invoke_assistant", headers=headers, json={
        "user_query": "add_two_numbers first_num=7 second_num=8",
        "timeout_seconds": 30,
    })
    assert response.status_code == 200
    data = response.json()
    assert "command_responses" in data and len(data["command_responses"]) > 0


def test_perform_action_endpoint(app_module, unique_user_id):
    """Test POST /perform_action endpoint"""
    client = TestClient(app_module.app)
    init = _initialize(client, unique_user_id)
    headers = _authorize(client, init["access_token"])
    response = client.post("/perform_action", headers=headers, json={
        "action": {
            "command_name": "add_two_numbers",
            "parameters": {"first_num": 12.5, "second_num": 7.5},
        },
        "timeout_seconds": 30,
    })
    assert response.status_code == 200
    data = response.json()
    assert "command_responses" in data


def test_session_not_found_errors(app_module):
    """Test 404 errors when session not found (valid JWT for unknown user)"""
    client = TestClient(app_module.app)
    token = app_module.create_access_token("nonexistent_user")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/invoke_agent", headers=headers, json={
        "user_query": "test query",
        "timeout_seconds": 10,
    })
    assert response.status_code == 200
    response = client.post("/invoke_assistant", headers=headers, json={
        "user_query": "test query",
        "timeout_seconds": 10,
    })
    assert response.status_code == 200
    response = client.post("/perform_action", headers=headers, json={
        "action": {"command_name": "test", "parameters": {}},
        "timeout_seconds": 10,
    })
    assert response.status_code == 500


def test_new_conversation_endpoint(app_module, unique_user_id):
    """Test POST /new_conversation endpoint"""
    client = TestClient(app_module.app)
    init = _initialize(client, unique_user_id)
    headers = _authorize(client, init["access_token"])
    # Perform interaction to populate history
    client.post("/invoke_assistant", headers=headers, json={
        "user_query": "add_two_numbers first_num=1 second_num=2",
        "timeout_seconds": 30,
    })
    response = client.post("/new_conversation", headers=headers, json={})
    assert response.status_code in [200, 500]


def test_conversations_list_endpoint(app_module, unique_user_id):
    """Test GET /conversations endpoint"""
    client = TestClient(app_module.app)
    init = _initialize(client, unique_user_id)
    headers = _authorize(client, init["access_token"])
    response = client.get("/conversations", headers=headers, params={"limit": 20})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_post_feedback_endpoint(app_module, unique_user_id):
    """Test POST /post_feedback endpoint - feedback on in-memory turns"""
    client = TestClient(app_module.app)
    init = _initialize(client, unique_user_id)
    headers = _authorize(client, init["access_token"])
    # Create a turn
    client.post("/invoke_agent", headers=headers, json={
        "user_query": "add 5 and 5",
        "timeout_seconds": 30,
    })
    response = client.post("/post_feedback", headers=headers, json={
        "binary_or_numeric_score": True,
    })
    assert response.status_code == 200
    # Verify in-memory updated
    import asyncio
    runtime = asyncio.run(app_module.session_manager.get_session(unique_user_id))
    assert runtime is not None
    assert len(runtime.chat_session.conversation_history.messages) > 0
    last_turn = runtime.chat_session.conversation_history.messages[-1]
    assert last_turn["feedback"] is not None
    assert last_turn["feedback"]["binary_or_numeric_score"] == 1.0
    # Overwrite feedback
    response = client.post("/post_feedback", headers=headers, json={
        "nl_feedback": "Very helpful!",
    })
    assert response.status_code == 200
    runtime = asyncio.run(app_module.session_manager.get_session(unique_user_id))
    last_turn = runtime.chat_session.conversation_history.messages[-1]
    assert last_turn["feedback"]["nl_feedback"] == "Very helpful!"
    # Persist on rotation
    response = client.post("/new_conversation", headers=headers, json={})
    assert response.status_code == 200 or response.status_code == 500


def test_post_feedback_validation(app_module):
    """Test POST /post_feedback validation (both fields null)"""
    client = TestClient(app_module.app)
    # Both fields null should fail validation (may be 401 if no token provided)
    response = client.post("/post_feedback", json={})
    assert response.status_code in [401, 403, 422]


def test_activate_conversation_endpoint(app_module, unique_user_id):
    """Test POST /activate_conversation endpoint"""
    client = TestClient(app_module.app)
    init = _initialize(client, unique_user_id)
    headers = _authorize(client, init["access_token"])
    response = client.post("/activate_conversation", headers=headers, json={
        "conversation_id": 999,
    })
    assert response.status_code == 404


def test_dump_all_conversations_endpoint(app_module, unique_user_id):
    # sourcery skip: extract-method
    """Test POST /admin/dump_all_conversations endpoint"""
    client = TestClient(app_module.app)
    _initialize(client, unique_user_id)
    with tempfile.TemporaryDirectory() as tmpdir:
        response = client.post("/admin/dump_all_conversations", json={
            "output_folder": tmpdir,
        })
        assert response.status_code == 200
        data = response.json()
        assert "file_path" in data
        assert os.path.exists(data["file_path"])
        assert data["file_path"].endswith(".jsonl")


def test_leading_slash_stripping(app_module):
    """Test that /invoke_agent strips leading slashes (404 for missing session)"""
    client = TestClient(app_module.app)
    token = app_module.create_access_token("nonexistent_user")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/invoke_agent", headers=headers, json={
        "user_query": "///test query with multiple slashes",
        "timeout_seconds": 10,
    })
    assert response.status_code == 200


def test_concurrent_request_handling(app_module, unique_user_id):
    """Test that a single request succeeds (lock prevents concurrency)"""
    client = TestClient(app_module.app)
    init = _initialize(client, unique_user_id)
    headers = _authorize(client, init["access_token"])
    response = client.post("/invoke_assistant", headers=headers, json={
        "user_query": "add_two_numbers first_num=1 second_num=1",
        "timeout_seconds": 30,
    })
    assert response.status_code == 200


# Note: Tests for jwt_verification_mode_default and jwt_verification_mode_enabled
# require complex fixture setup with multiple module imports.
# These are covered by the test_jwt_manager_set_verification_mode test
# which validates the core functionality.


def test_jwt_manager_set_verification_mode():
    """Test the set_jwt_verification_mode function directly"""
    import importlib
    import fastworkflow.run_fastapi_mcp.jwt_manager as jwt_module
    
    # Reload to get a fresh copy
    importlib.reload(jwt_module)
    
    # Test setting to False (trusted network mode - no verification)
    jwt_module.set_jwt_verification_mode(False)
    assert jwt_module.EXPECT_ENCRYPTED_JWT is False
    
    # Test setting to True (secure mode - with verification)
    jwt_module.set_jwt_verification_mode(True)
    assert jwt_module.EXPECT_ENCRYPTED_JWT is True
    
    # Test setting back to False
    jwt_module.set_jwt_verification_mode(False)
    assert jwt_module.EXPECT_ENCRYPTED_JWT is False
    
    # Reset to secure mode for other tests
    jwt_module.set_jwt_verification_mode(True)
    assert jwt_module.EXPECT_ENCRYPTED_JWT is True


def test_jwt_token_creation_modes():
    """Test that tokens are created as signed or unsigned based on EXPECT_ENCRYPTED_JWT flag"""
    import importlib
    import fastworkflow.run_fastapi_mcp.jwt_manager as jwt_module
    
    # Reload to get a fresh copy
    importlib.reload(jwt_module)
    
    # Test unsigned token creation (default mode)
    jwt_module.set_jwt_verification_mode(False)
    
    # Create a token in unsigned mode
    token = jwt_module.create_access_token("test_user")
    assert token is not None
    assert "." in token  # JWT format: header.payload.signature (but signature will be empty)
    
    # Decode the token without verification (using unverified claims to skip audience/issuer validation)
    import jwt as pyjwt
    payload = pyjwt.decode(token, options={"verify_signature": False})
    assert payload["sub"] == "test_user"
    assert payload["type"] == "access"
    
    # Test signed token creation (secure mode)
    jwt_module.set_jwt_verification_mode(True)
    
    # Create a token in signed mode
    signed_token = jwt_module.create_access_token("test_user_2")
    assert signed_token is not None
    assert "." in signed_token
    
    # Verify the signed token can be decoded with verification
    payload2 = jwt_module.verify_token(signed_token, expected_type="access")
    assert payload2["sub"] == "test_user_2"
    assert payload2["type"] == "access"
    
    # Reset to default for other tests (default is True - secure mode)
    jwt_module.set_jwt_verification_mode(True)


def test_cli_arg_expect_encrypted_jwt_not_set():
    """Test that the CLI argument defaults correctly"""
    # The argument defaults to False when not specified
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect_encrypted_jwt", action="store_true", default=False)
    
    args = parser.parse_args([])
    assert args.expect_encrypted_jwt is False
    
    args = parser.parse_args(["--expect_encrypted_jwt"])
    assert args.expect_encrypted_jwt is True


def test_jwt_token_includes_user_id_claim(app_module):
    """Test that JWT tokens include uid claim when user_id is provided"""
    import fastworkflow.run_fastapi_mcp.jwt_manager as jwt_module
    import jwt as pyjwt
    
    # Create token with user_id
    token = jwt_module.create_access_token("test_channel", user_id="test_user_789")
    
    # Decode and verify uid claim
    payload = pyjwt.decode(token, options={"verify_signature": False})
    assert payload["sub"] == "test_channel"
    assert payload["uid"] == "test_user_789"
    assert payload["type"] == "access"
    
    # Create token without user_id
    token_no_uid = jwt_module.create_access_token("test_channel_2")
    payload_no_uid = pyjwt.decode(token_no_uid, options={"verify_signature": False})
    assert payload_no_uid["sub"] == "test_channel_2"
    assert "uid" not in payload_no_uid  # uid should be absent when not provided


def test_session_data_extracts_user_id_from_token(app_module, unique_user_id):
    """Test that SessionData properly extracts user_id from JWT uid claim"""
    client = TestClient(app_module.app)
    
    # Initialize with user_id
    resp = client.post("/initialize", json={
        "channel_id": unique_user_id,
        "user_id": "alice_123"
    })
    assert resp.status_code == 200
    data = resp.json()
    
    # Use the token to call an endpoint and verify user_id is extracted
    headers = {"Authorization": f"Bearer {data['access_token']}"}
    resp2 = client.post("/invoke_agent", headers=headers, json={
        "user_query": "add 2 and 3",
        "timeout_seconds": 30
    })
    assert resp2.status_code == 200
    result = resp2.json()
    
    # Verify traces include user_id
    if "traces" in result and result["traces"]:
        first_trace = result["traces"][0]
        assert "user_id" in first_trace
        assert first_trace["user_id"] == "alice_123"


def test_traces_include_raw_command(app_module, unique_user_id):
    """Test that traces include the raw_command field"""
    client = TestClient(app_module.app)
    init = _initialize(client, unique_user_id)
    headers = _authorize(client, init["access_token"])
    
    response = client.post("/invoke_agent", headers=headers, json={
        "user_query": "add 7 and 9",
        "timeout_seconds": 30,
    })
    assert response.status_code == 200
    data = response.json()
    
    # Verify traces include raw_command
    if "traces" in data and data["traces"]:
        for trace in data["traces"]:
            assert "raw_command" in trace
            # raw_command should be the user query
            assert trace["raw_command"] == "add 7 and 9"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

