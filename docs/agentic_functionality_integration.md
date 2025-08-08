# Specification for Integrating Agentic Functionality into fastWorkflow Core

## 1. Overview

### 1.1 Purpose
This specification outlines the integration of agentic (AI-driven, autonomous decision-making) capabilities directly into the core fastWorkflow runtime, eliminating the need for separate `run` and `run_agent` programs. The goal is to unify the execution model under a single entry point (`run/__main__.py`) while allowing users to switch between assistant mode (direct command processing) and agent mode (autonomous planning and execution using DSPy-based agents).

The integration introduces a new `run_as` parameter to `fastworkflow.ChatSession`, with values `"RUN_AS_ASSISTANT"` (default, for traditional imperative command execution) and `"RUN_AS_AGENT"` (for agentic, goal-oriented processing). In agent mode, the system treats the entire workflow as a "single complex tool" with a dynamic description that evolves based on context changes, leveraging GenAI-generated metadata, command details, and runtime introspection.

This enhances usability by enabling natural language interactions that autonomously handle command selection, parameter resolution, context navigation, and error recovery, while falling back to user input when necessary.

### 1.2 Scope
- **In Scope**:
  - Modifications to `ChatSession` for mode-based behavior.
  - Refinements to core commands (e.g., "what_can_i_do") for richer metadata.
  - Embedding a DSPy-based "main agent" within fastWorkflow for agent mode.
  - Dynamic tool description generation using GenAI artifacts (e.g., workflow_description.json from post-build step).
  - Context-aware agent logic for detecting changes and refreshing command info.
  - Integration with upcoming features like command dependency graphs for parameter resolution suggestions.
- **Out of Scope**:
  - Full runtime execution of dependency chains (covered in command_dependency_graph.md spec; agent will suggest but not auto-execute).
  - Cross-workflow agentic behavior.
  - Advanced agent training/optimization beyond basic DSPy ReAct setup.
  - UI changes beyond CLI prompts.

### 1.3 Key Benefits
- **Unified Runtime**: Single codebase for both modes, reducing maintenance overhead.
- **Dynamic Adaptability**: Agent treats workflow as a evolving tool, querying runtime state (e.g., current context) to update its "tool description."
- **Improved Error Handling**: Autonomous recovery from missing parameters using dependency clues; graceful fallback to user queries.
- **Enhanced Discoverability**: Richer "what_can_i_do" responses provide comprehensive command details for both modes.
- **Leverage Existing Specs**: Builds on GenAI post-processing (build_postprocessing_using_dspy.md) for metadata and command_dependency_graph.md for suggestions.

### 1.4 Assumptions and Dependencies
- **Dependencies**:
  - DSPy library for agent implementation (already in run_agent).
  - GenAI post-processing phase (from build_postprocessing_using_dspy.md) generates artifacts like `workflow_description.json`, command docstrings, and field metadata.
  - Command dependency graph (from command_dependency_graph.md) for parameter resolution suggestions.
  - Existing fastWorkflow components: `ChatSession`, `CommandExecutor`, core commands (e.g., "what_can_i_do", "go_up", "reset_context", "current_context").
- **Assumptions**:
  - Workflows are built with `_commands` directory structure, Pydantic-based Signatures, and context hierarchies defined in `context_inheritance_model.json` and `context_containment_model.json` (if available).
  - Environment variables (e.g., `LLM_AGENT`, API keys) are set for DSPy in agent mode.
  - Agent mode requires a trained workflow (checked via `tiny_ambiguous_threshold.json` as in run_agent).
  - Semantic matching uses embeddings (e.g., Sentence Transformers) for dynamic descriptions.

## 2. Architecture

### 2.1 High-Level Components
1. **ChatSession Enhancements**:
   - New `run_as` parameter: Enum-like string (`"RUN_AS_ASSISTANT"` or `"RUN_AS_AGENT"`).
   - Mode-specific processing loops in `ChatSession.start()`.

2. **Refined Core Commands**:
   - "what_can_i_do": Returns detailed JSON-structured info on available commands.

3. **Main Agent (DSPy ReAct)**:
   - Embedded in fastWorkflow; initialized in agent mode.
   - Views workflow as a single tool with dynamic description.

4. **Dynamic Tool Description Generator**:
   - Aggregates static (GenAI artifacts) and runtime (context queries) info.

5. **Error Handling and Fallback**:
   - Integrates dependency graph for suggestions; uses "AskUser" tool for unresolved cases.

### 2.2 Execution Flow
- **Initialization** (`ChatSession.__init__`):
  - Accept `run_as` (default: `"RUN_AS_ASSISTANT"`).
  - If `"RUN_AS_AGENT"`, validate workflow training and initialize main agent.

- **Startup** (`ChatSession.start()`):
  - Common: Load workflow, execute startup command/action.
  - Assistant Mode: Enter imperative command loop (as in current `run/__main__.py`).
  - Agent Mode: Enter agentic loop (natural language input → agent processing → output).

- **Agent Mode Loop**:
  1. Receive user query (natural language).
  2. Pass to main agent, which plans using "WorkflowTool" (dynamic description).
  3. Agent invokes tool (delegates to command execution).
  4. Process response: Detect context changes, refresh descriptions if needed.
  5. Handle errors: Use dependency clues or ask user.
  6. Output formatted response (as in run_agent).

- **Build Integration**:
  - During `fastworkflow build`, generate/update GenAI artifacts (descriptions, docstrings).
  - Ensure dependency graph is built for parameter suggestions.

### 2.3 DSPy Components
- **Main Agent Signature** (PlanningAgentSignature):
  ```
  class PlanningAgentSignature(dspy.Signature):
      \"\"\"Plan and execute steps to answer the user query using the WorkflowTool.\"\"\"
      user_query: str
      final_answer: str = dspy.OutputField(desc=\"Comprehensive response after workflow interactions.\")
  ```
- **WorkflowTool** (Single Tool for Agent):
  - Func: Delegates to command executor or sub-agents.
  - Dynamic `__doc__`: Generated description (see Section 3).

- **Sub-Tools** (Internal to WorkflowTool):
  - "AskUser": For clarifications (as in run_agent).
  - Core command wrappers (e.g., "what_can_i_do" for refresh).

- **Agent Configuration**:
  - ReAct with max_iters=25 (configurable).
  - LM: From `LLM_AGENT` env var.

## 3. Detailed Features

### 3.1 Refined "what_can_i_do" Command
- **Output Structure**: JSON for agent consumption; formatted text for assistant mode.
  - Context: `{name, description (from _<Context>.py docstring), inheritance: [parents], containment: [children]}`.
  - Commands: List of `{qualified_name, signature_docstring, inputs: [{name, type, description, examples}], plain_utterances: [str]}`.
- **Implementation**: Extend `CommandExecutor` to aggregate from `CommandDirectory`, GenAI docstrings, and Pydantic fields.
- **Usage**: Called automatically by agent on context changes; available in both modes.

### 3.2 Main Agent Implementation
- **Location**: New module `fastworkflow/agent_integration.py`.
- **Initialization**:
  - Load static data: `workflow_description.json`, context lists/hierarchies.
  - Query runtime: Initial "what_can_i_do" for core commands/current context.
- **Dynamic Tool Description**:
  - Base: Workflow overview + context hierarchies + core commands.
  - Per-Context: Append current context's "what_can_i_do" details.
  - Update: On context change detection (e.g., parse response for "Context changed to X"), call "what_can_i_do" and refresh `__doc__`.
- **Planning Logic**:
  - Use ReAct to break down query into steps, invoking WorkflowTool.
  - For commands: Delegate to `CommandExecutor`.

### 3.3 Error Handling and Fallback
- **Missing Parameters**:
  - Parse validation message (from command_dependency_graph.md).
  - If clues (e.g., suggested commands via graph query), attempt resolution autonomously.
  - Else: Invoke "AskUser" with validation message + request for values.
- **Context Change Detection**:
  - Post-response: Semantic check (e.g., embeddings) for phrases like "switched to context X".
  - If detected and not in history: Call "what_can_i_do"; update tool description.
- **Stopping Conditions**:
  - Stop tool calls if no resolution clues; fallback to user.

### 3.4 Core Commands in Agent
- Wrapper tools for: "go_up", "reset_context", "current_context", "what_can_i_do".
- Included in dynamic description with refined details.

## 4. Implementation Details
- **ChatSession Changes** (`fastworkflow/chat_session.py`):
  - Add `run_as` param; mode-specific loops in `start()`.
- **Agent Module** (`fastworkflow/agent_integration.py`):
  - Similar to run_agent/agent_module.py, but integrated.
- **Build Enhancements** (`build/__main__.py`):
  - Ensure GenAI post-processing generates required JSON artifacts.
- **Performance**: Cache dynamic descriptions; limit agent iterations.
- **Error Handling**: Fallback to assistant mode on agent failures.

## 5. Testing and Validation
- **Unit Tests**: Core command refinements, dynamic description generation.
- **Integration Tests**: End-to-end agent mode with sample workflows.
- **Edge Cases**: Context switches, missing params without clues, untrained workflows.

This spec unifies fastWorkflow's runtime while enhancing agentic capabilities, leveraging existing and upcoming features for a robust system. 