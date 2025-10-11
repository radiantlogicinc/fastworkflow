#!/usr/bin/env python3
"""
Quick test script for the /invoke_agent_stream endpoint
Tests SSE streaming functionality with FastWorkflow
"""

import json
import requests
import time

BASE_URL = "http://localhost:8000"

def parse_sse_events(response_text):
    """Parse SSE event stream into structured events"""
    events = []
    current_event = {}
    
    for line in response_text.split('\n'):
        if line.startswith('event: '):
            current_event['event'] = line[7:]
        elif line.startswith('data: '):
            current_event['data'] = json.loads(line[6:])
        elif line == '':
            if current_event:
                events.append(current_event)
                current_event = {}
    
    return events

def test_streaming_endpoint():
    """Test the streaming endpoint with a simple workflow"""
    
    # Note: This test requires:
    # 1. FastAPI server running (uvicorn services.run_fastapi.main:app)
    # 2. A valid workflow at the specified path
    # 3. Required environment files
    
    user_id = f"test_user_{int(time.time())}"
    
    # Step 1: Initialize session
    print(f"1. Initializing session for {user_id}...")
    init_payload = {
        "user_id": user_id,
        "workflow_path": "/home/drawal/rl/fastworkflow/fastworkflow/examples/todo_list",
        "env_file_path": "/home/drawal/rl/fastworkflow/fastworkflow/examples/fastworkflow.env",
        "passwords_file_path": "/home/drawal/rl/fastworkflow/fastworkflow/examples/fastworkflow.passwords.env",
        "show_agent_traces": True
    }
    
    try:
        init_response = requests.post(f"{BASE_URL}/initialize", json=init_payload)
        init_response.raise_for_status()
        print(f"   ✓ Session initialized: {init_response.json()}")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return
    
    # Step 2: Test streaming endpoint
    print(f"\n2. Testing /invoke_agent_stream...")
    stream_payload = {
        "user_id": user_id,
        "user_query": "create a todo item: test streaming endpoint",
        "timeout_seconds": 30
    }
    
    try:
        # Stream the response
        response = requests.post(
            f"{BASE_URL}/invoke_agent_stream",
            json=stream_payload,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        response.raise_for_status()
        
        print("   Streaming events:")
        trace_count = 0
        command_output = None
        error_occurred = False
        
        # Process SSE events
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('event: '):
                event_type = line[7:]
                print(f"   → {event_type}")
                
                if event_type == 'trace':
                    trace_count += 1
                elif event_type == 'error':
                    error_occurred = True
            
            elif line.startswith('data: '):
                data = json.loads(line[6:])
                
                if 'command_name' in data:  # This is a command_output
                    command_output = data
                    print(f"   ✓ Command output received: {data.get('command_name')}")
                elif 'detail' in data:  # This is an error
                    print(f"   ✗ Error: {data['detail']}")
        
        # Validate results
        print(f"\n3. Validation:")
        print(f"   Trace events received: {trace_count}")
        print(f"   Command output: {'✓' if command_output else '✗'}")
        print(f"   Errors: {'✗' if error_occurred else '✓ None'}")
        
        if command_output:
            print(f"   Success: {command_output.get('success')}")
            print(f"   Workflow: {command_output.get('workflow_name')}")
        
        return not error_occurred and command_output is not None
        
    except Exception as e:
        print(f"   ✗ Failed to stream: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FastWorkflow /invoke_agent_stream endpoint")
    print("=" * 60)
    
    success = test_streaming_endpoint()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Test PASSED")
    else:
        print("✗ Test FAILED")
    print("=" * 60)

