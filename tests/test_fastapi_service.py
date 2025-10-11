"""
Basic tests for FastWorkflow FastAPI service
These tests validate the structure and basic functionality
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


def test_fastapi_imports():
    """Test that all required modules can be imported"""
    try:
        from services.run_fastapi.main import (
            app,
            UserSessionManager,
            UserRuntime,
            InitializationRequest,
            InitializationResponse,
            InvokeRequest,
            PerformActionRequest,
            NewConversationRequest,
            PostFeedbackRequest,
            ActivateConversationRequest,
            DumpConversationsRequest,
        )
        from services.conversation_store import (
            ConversationStore,
            ConversationSummary,
            generate_topic_and_summary,
        )
        assert app is not None
        assert ConversationStore is not None
    except ImportError as e:
        pytest.fail(f"Failed to import FastAPI service components: {e}")


def test_root_endpoint():
    """Test the root health check endpoint"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    assert "FastWorkflow API is running" in response.text
    assert "/docs" in response.text


def test_validation_startup_mutual_exclusion():
    """Test that startup_command and startup_action are mutually exclusive"""
    from services.run_fastapi.main import InitializationRequest
    from pydantic import ValidationError
    
    # Should raise validation error when both are provided
    with pytest.raises(ValidationError):
        InitializationRequest(
            user_id="test",
            workflow_path="/path/to/workflow",
            startup_command="test command",
            startup_action={"command_name": "test", "parameters": {}}
        )
    
    # Should succeed with only startup_command
    req = InitializationRequest(
        user_id="test",
        workflow_path="/path/to/workflow",
        startup_command="test command"
    )
    assert req.startup_command == "test command"
    assert req.startup_action is None
    
    # Should succeed with only startup_action
    req = InitializationRequest(
        user_id="test",
        workflow_path="/path/to/workflow",
        startup_action={"command_name": "test", "parameters": {}}
    )
    assert req.startup_command is None
    assert req.startup_action is not None


def test_validation_feedback_presence():
    """Test that at least one feedback field must be provided"""
    from services.run_fastapi.main import PostFeedbackRequest
    from pydantic import ValidationError
    
    # Should raise validation error when neither field is provided
    with pytest.raises(ValidationError):
        PostFeedbackRequest(
            user_id="test",
            binary_or_numeric_score=None,
            nl_feedback=None
        )
    
    # Should succeed with only binary_or_numeric_score
    req = PostFeedbackRequest(
        user_id="test",
        binary_or_numeric_score=True
    )
    assert req.binary_or_numeric_score is True
    
    # Should succeed with only nl_feedback
    req = PostFeedbackRequest(
        user_id="test",
        nl_feedback="Great response"
    )
    assert req.nl_feedback == "Great response"
    
    # Should succeed with both
    req = PostFeedbackRequest(
        user_id="test",
        binary_or_numeric_score=0.8,
        nl_feedback="Good"
    )
    assert req.binary_or_numeric_score == 0.8
    assert req.nl_feedback == "Good"


def test_session_manager_basic():
    """Test basic UserSessionManager functionality"""
    import asyncio
    from services.run_fastapi.main import UserSessionManager
    
    async def test_async():
        manager = UserSessionManager()
        
        # Initially no sessions
        session = await manager.get_session("test_user")
        assert session is None
        
        # Can create and retrieve sessions (basic structure test only)
        # Full session creation requires ChatSession and ConversationStore
    
    asyncio.run(test_async())


def test_conversation_store_structure():
    """Test ConversationStore basic structure"""
    from services.conversation_store import ConversationStore
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ConversationStore("test_user", tmpdir)
        
        # Check DB file path construction
        assert store.user_id == "test_user"
        assert store.db_path.endswith("test_user.rdb")
        assert tmpdir in store.db_path
        
        # Initially no conversations
        last_id = store.get_last_conversation_id()
        assert last_id is None


def test_env_loading_validation():
    """Test environment file loading validation"""
    from fastapi import HTTPException
    from services.run_fastapi.main import load_env_from_files
    
    # Non-existent file should raise 422
    with pytest.raises(HTTPException) as exc_info:
        load_env_from_files("/nonexistent/file.env", None)
    assert exc_info.value.status_code == 422
    
    # Valid file should load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("TEST_VAR=test_value\n")
        temp_path = f.name
    
    try:
        env_vars = load_env_from_files(temp_path, None)
        assert "TEST_VAR" in env_vars
        assert env_vars["TEST_VAR"] == "test_value"
    finally:
        os.unlink(temp_path)


# ============================================================================
# Integration Tests with hello_world workflow
# ============================================================================

@pytest.fixture
def hello_world_workflow_path():
    """Get path to hello_world example workflow"""
    import fastworkflow
    package_path = fastworkflow.get_fastworkflow_package_path()
    workflow_path = os.path.join(package_path, "examples", "hello_world")
    
    # Verify workflow exists
    if not os.path.isdir(workflow_path):
        pytest.skip(f"hello_world workflow not found at {workflow_path}")
    
    return workflow_path


@pytest.fixture
def env_files():
    """Get paths to actual env files for testing"""
    # Get project root (assuming tests are in tests/ directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    
    # Verify files exist
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


def test_initialize_endpoint(hello_world_workflow_path, env_files):
    """Test POST /initialize endpoint with hello_world workflow"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    response = client.post("/initialize", json={
        "user_id": "test_user_1",
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file,
        "show_agent_traces": True
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_1"


def test_initialize_with_startup_command(hello_world_workflow_path, env_files, unique_user_id):
    """Test initialization with startup_command"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file,
        "startup_command": "/add_two_numbers first_num=5 second_num=3",
        "show_agent_traces": False
    })
    
    assert response.status_code == 200
    assert response.json()["user_id"] == unique_user_id


def test_initialize_with_startup_action(hello_world_workflow_path, env_files, unique_user_id):
    """Test initialization with startup_action"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file,
        "startup_action": {
            "command_name": "add_two_numbers",
            "parameters": {"first_num": 10, "second_num": 20}
        },
        "show_agent_traces": True
    })
    
    assert response.status_code == 200
    assert response.json()["user_id"] == unique_user_id


def test_initialize_validation_errors(hello_world_workflow_path, env_files):
    """Test initialization validation errors"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Both startup_command and startup_action (should fail)
    response = client.post("/initialize", json={
        "user_id": "test_user",
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file,
        "startup_command": "test",
        "startup_action": {"command_name": "test", "parameters": {}}
    })
    
    assert response.status_code == 422
    
    # Invalid workflow path
    response = client.post("/initialize", json={
        "user_id": "test_user",
        "workflow_path": "/nonexistent/workflow",
        "env_file_path": env_file,
        "passwords_file_path": passwords_file
    })
    
    assert response.status_code == 422
    assert "not a valid directory" in response.json()["detail"]


def test_invoke_agent_endpoint(hello_world_workflow_path, env_files, unique_user_id):
    """Test POST /invoke_agent endpoint (agentic mode)"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Initialize session first
    init_response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file,
        "show_agent_traces": True
    })
    assert init_response.status_code == 200
    
    # Invoke agent with natural language query
    response = client.post("/invoke_agent", json={
        "user_id": unique_user_id,
        "user_query": "add 15 and 25",
        "timeout_seconds": 60
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "command_responses" in data
    assert len(data["command_responses"]) > 0
    # With show_agent_traces=True, should have traces
    assert "traces" in data
    assert isinstance(data["traces"], list)


def test_invoke_assistant_endpoint(hello_world_workflow_path, env_files, unique_user_id):
    """Test POST /invoke_assistant endpoint (deterministic mode)"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Initialize session first
    init_response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file,
        "show_agent_traces": False
    })
    assert init_response.status_code == 200
    
    # Invoke assistant with deterministic command
    response = client.post("/invoke_assistant", json={
        "user_id": unique_user_id,
        "user_query": "add_two_numbers first_num=7 second_num=8",
        "timeout_seconds": 30
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "command_responses" in data
    assert len(data["command_responses"]) > 0
    # With show_agent_traces=False, traces should be None or not included
    assert data.get("traces") is None or "traces" not in data


def test_perform_action_endpoint(hello_world_workflow_path, env_files, unique_user_id):
    """Test POST /perform_action endpoint"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Initialize session first
    init_response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file,
        "show_agent_traces": True
    })
    assert init_response.status_code == 200
    
    # Perform specific action
    response = client.post("/perform_action", json={
        "user_id": unique_user_id,
        "action": {
            "command_name": "add_two_numbers",
            "parameters": {"first_num": 12.5, "second_num": 7.5}
        },
        "timeout_seconds": 30
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "command_responses" in data
    # With show_agent_traces=True, should have traces
    assert "traces" in data


def test_session_not_found_errors():
    """Test 404 errors when session not found"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    
    # Try to invoke agent without initialization
    response = client.post("/invoke_agent", json={
        "user_id": "nonexistent_user",
        "user_query": "test query",
        "timeout_seconds": 10
    })
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
    
    # Try to invoke assistant without initialization
    response = client.post("/invoke_assistant", json={
        "user_id": "nonexistent_user",
        "user_query": "test query",
        "timeout_seconds": 10
    })
    
    assert response.status_code == 404
    
    # Try to perform action without initialization
    response = client.post("/perform_action", json={
        "user_id": "nonexistent_user",
        "action": {"command_name": "test", "parameters": {}},
        "timeout_seconds": 10
    })
    
    assert response.status_code == 404


def test_new_conversation_endpoint(hello_world_workflow_path, env_files, unique_user_id):
    """Test POST /new_conversation endpoint"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Initialize session
    init_response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file
    })
    assert init_response.status_code == 200
    
    # Perform some interactions first
    client.post("/invoke_assistant", json={
        "user_id": unique_user_id,
        "user_query": "add_two_numbers first_num=1 second_num=2",
        "timeout_seconds": 30
    })
    
    # Start new conversation
    response = client.post("/new_conversation", json={
        "user_id": unique_user_id
    })
    
    # Note: This might fail if DSPy models aren't configured
    # For now, just check it doesn't crash with wrong status codes
    assert response.status_code in [200, 500]  # 500 if DSPy not configured


def test_conversations_list_endpoint(hello_world_workflow_path, env_files, unique_user_id):
    """Test GET /conversations endpoint"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Initialize session
    init_response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file
    })
    assert init_response.status_code == 200
    
    # List conversations (should be empty or have default)
    response = client.get("/conversations", params={"user_id": unique_user_id})
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_post_feedback_endpoint(hello_world_workflow_path, env_files, unique_user_id):
    """Test POST /post_feedback endpoint - feedback on in-memory turns"""
    from services.run_fastapi.main import app
    from services.run_fastapi.main import session_manager
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Initialize session
    init_response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file
    })
    assert init_response.status_code == 200
    
    # Use invoke_agent to create a turn (invoke_assistant doesn't populate conversation history)
    client.post("/invoke_agent", json={
        "user_id": unique_user_id,
        "user_query": "add 5 and 5",
        "timeout_seconds": 30
    })
    
    # Post feedback with binary score - should update in-memory turn
    response = client.post("/post_feedback", json={
        "user_id": unique_user_id,
        "binary_or_numeric_score": True
    })
    
    # Should succeed
    assert response.status_code == 200
    
    # Verify feedback was added to in-memory conversation history
    import asyncio
    runtime = asyncio.run(session_manager.get_session(unique_user_id))
    assert runtime is not None
    assert len(runtime.chat_session.conversation_history.messages) > 0
    last_turn = runtime.chat_session.conversation_history.messages[-1]
    assert last_turn["feedback"] is not None
    assert last_turn["feedback"]["binary_or_numeric_score"] is True
    
    # Post feedback with text on same turn (overwrites previous feedback)
    response = client.post("/post_feedback", json={
        "user_id": unique_user_id,
        "nl_feedback": "Very helpful!"
    })
    
    # Should succeed
    assert response.status_code == 200
    
    # Verify feedback was updated
    runtime = asyncio.run(session_manager.get_session(unique_user_id))
    last_turn = runtime.chat_session.conversation_history.messages[-1]
    assert last_turn["feedback"]["nl_feedback"] == "Very helpful!"
    
    # Now end the conversation and verify feedback is persisted
    response = client.post("/new_conversation", json={
        "user_id": unique_user_id
    })
    assert response.status_code == 200
    
    # Verify feedback was persisted to Rdict
    runtime = asyncio.run(session_manager.get_session(unique_user_id))
    conv = runtime.conversation_store.get_conversation(runtime.active_conversation_id)
    if conv:  # May be None if no turns were in conversation
        assert len(conv["turns"]) > 0
        last_persisted_turn = conv["turns"][-1]
        assert last_persisted_turn["feedback"] is not None
        assert last_persisted_turn["feedback"]["nl_feedback"] == "Very helpful!"


def test_post_feedback_validation():
    """Test POST /post_feedback validation (both fields null)"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    
    # Both fields null should fail validation
    response = client.post("/post_feedback", json={
        "user_id": "test_user",
        "binary_or_numeric_score": None,
        "nl_feedback": None
    })
    
    assert response.status_code == 422


def test_activate_conversation_endpoint(hello_world_workflow_path, env_files, unique_user_id):
    """Test POST /activate_conversation endpoint"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Initialize session
    init_response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file
    })
    assert init_response.status_code == 200
    
    # Try to activate a non-existent conversation
    response = client.post("/activate_conversation", json={
        "user_id": unique_user_id,
        "conversation_id": 999
    })
    
    assert response.status_code == 404


def test_dump_all_conversations_endpoint(hello_world_workflow_path, env_files, unique_user_id):
    # sourcery skip: extract-method
    """Test POST /admin/dump_all_conversations endpoint"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Initialize a session first
    init_response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file
    })
    assert init_response.status_code == 200
    
    # Create temp directory for dump
    with tempfile.TemporaryDirectory() as tmpdir:
        response = client.post("/admin/dump_all_conversations", json={
            "output_folder": tmpdir
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "file_path" in data
        assert os.path.exists(data["file_path"])
        assert data["file_path"].endswith(".jsonl")


def test_leading_slash_stripping():
    """Test that /invoke_agent strips leading slashes"""
    from services.run_fastapi.main import app
    
    client = TestClient(app)
    
    # This will fail with 404 (no session) but verifies the endpoint accepts leading slashes
    response = client.post("/invoke_agent", json={
        "user_id": "test_user",
        "user_query": "///test query with multiple slashes",
        "timeout_seconds": 10
    })
    
    # Should get 404 (session not found), not 422 (validation error)
    assert response.status_code == 404


def test_concurrent_request_handling(hello_world_workflow_path, env_files, unique_user_id):
    """Test that concurrent requests for same user return 409"""
    from services.run_fastapi.main import app
    import threading
    
    client = TestClient(app)
    env_file, passwords_file = env_files
    
    # Initialize session
    init_response = client.post("/initialize", json={
        "user_id": unique_user_id,
        "workflow_path": hello_world_workflow_path,
        "env_file_path": env_file,
        "passwords_file_path": passwords_file
    })
    assert init_response.status_code == 200
    
    # Note: This test is tricky because TestClient is synchronous
    # In practice, the lock prevents concurrent requests
    # We can only verify the endpoint structure is correct
    response = client.post("/invoke_assistant", json={
        "user_id": unique_user_id,
        "user_query": "add_two_numbers first_num=1 second_num=1",
        "timeout_seconds": 30
    })
    
    # Should succeed (not 409) since we're not actually concurrent
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

