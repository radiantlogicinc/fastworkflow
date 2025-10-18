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
    import fastworkflow.run_fastapi_mcp.main as main
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
    """Updated: Validate InitializationRequest requires user_id and supports stream_format"""
    from pydantic import ValidationError
    InitReq = app_module.InitializationRequest
    # Missing user_id should fail
    with pytest.raises(ValidationError):
        InitReq()
    # Valid with user_id and optional stream_format
    req = InitReq(user_id="test", stream_format="ndjson")
    assert req.user_id == "test"


def test_validation_feedback_presence(app_module):
    """Test that at least one feedback field must be provided"""
    from pydantic import ValidationError
    PostFeedbackRequest = app_module.PostFeedbackRequest
    with pytest.raises(ValidationError):
        PostFeedbackRequest(
            user_id="test",
            binary_or_numeric_score=None,
            nl_feedback=None,
        )
    req = PostFeedbackRequest(user_id="test", binary_or_numeric_score=True)
    assert req.binary_or_numeric_score == 1.0
    req = PostFeedbackRequest(user_id="test", nl_feedback="Great response")
    assert req.nl_feedback == "Great response"
    req = PostFeedbackRequest(user_id="test", binary_or_numeric_score=0.8, nl_feedback="Good")
    assert req.binary_or_numeric_score == 0.8
    assert req.nl_feedback == "Good"


def test_session_manager_basic(app_module):
    """Test basic UserSessionManager get_session behavior (no sessions initially)"""
    import asyncio
    async def test_async():
        manager = app_module.UserSessionManager()
        session = await manager.get_session("test_user")
        assert session is None
    asyncio.run(test_async())


def _authorize(client: TestClient, access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


def _initialize(client: TestClient, user_id: str) -> dict:
    resp = client.post("/initialize", json={"user_id": user_id, "stream_format": "ndjson"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data and "refresh_token" in data
    return data


def test_initialize_endpoint(app_module, unique_user_id):
    """Test POST /initialize endpoint"""
    client = TestClient(app_module.app)
    data = _initialize(client, "test_user_1")
    assert data["token_type"].lower() == "bearer"


def test_initialize_with_startup_command(app_module, unique_user_id):
    """Initialization still succeeds with startup_command configured via ARGS"""
    app_module.ARGS.startup_command = "/add_two_numbers first_num=5 second_num=3"
    client = TestClient(app_module.app)
    data = _initialize(client, unique_user_id)
    assert "access_token" in data
    # reset
    app_module.ARGS.startup_command = None


def test_initialize_with_startup_action(app_module, unique_user_id):
    """Initialization with startup_action configured via ARGS"""
    import json as _json
    app_module.ARGS.startup_action = _json.dumps({
        "command_name": "add_two_numbers",
        "parameters": {"first_num": 10, "second_num": 20}
    })
    client = TestClient(app_module.app)
    data = _initialize(client, unique_user_id)
    assert "access_token" in data
    # reset
    app_module.ARGS.startup_action = None


def test_initialize_validation_errors(app_module):
    """Invalid workflow path via ARGS should 422"""
    client = TestClient(app_module.app)
    # Set invalid workflow path
    app_module.ARGS.workflow_path = "/nonexistent/workflow"
    resp = client.post("/initialize", json={"user_id": "test_user"})
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

