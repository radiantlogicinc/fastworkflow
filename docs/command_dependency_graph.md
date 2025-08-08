# Specification for Generating Command Parameter Dependency Graphs in fastWorkflow

## 1. Introduction

### 1.1 Purpose
In the fastWorkflow framework, commands are defined with `Signature` classes that include `Input` and `Output` Pydantic models. These models specify the parameters required to invoke a command (`Input`) and the results produced by the command (`Output`). A common challenge in workflows is handling missing input parameters for a command Y, which may be obtainable by invoking another command X whose output matches Y's required input.

This specification defines a system for automatically generating a **dependency graph** where:
- **Nodes** represent commands (identified by their qualified names, e.g., `ChatRoom/add_user`).
- **Directed edges** from Y to X indicate that command Y *depends on* command X, meaning X's output can satisfy one or more of Y's input parameters.

The primary use case is handling validation errors for missing input parameters in command Y. The graph enables quick querying of suggested commands [X] that can provide the missing value. For example:
- Error: "Missing value for Y.input_param".
- Query: Get list of X's whose outputs match Y.input_param.

Graph generation occurs automatically during every build phase (via `fastworkflow build`), producing a JSON artifact (e.g., `___command_info/parameter_dependency_graph.json`).

### 1.2 Scope
- **In Scope**: Matching based on parameter metadata within a single workflow's commands. Supports both exact and semantic matching with configurable thresholds. Includes a runtime function to query suggestions for missing parameters.
- **Out of Scope**: Runtime execution of dependencies (covered in a separate spec); cross-workflow dependencies; handling of dynamic/runtime-generated parameters.

### 1.3 Assumptions
- All commands are defined in the `_commands` directory with `Signature` classes containing `Input` and `Output` Pydantic models (as per fastWorkflow's command structure).
- Parameter types are inferred from Pydantic field annotations (e.g., `str`, `int`, `List[str]`; serialized as strings for matching).
- Semantic search relies on embeddings (e.g., via Sentence Transformers or a similar library integrated into fastWorkflow).
- The graph is static and regenerated on every build.
- Metadata access uses `CommandDirectory` and `RoutingDefinition` instead of direct file access.

## 2. Definitions

### 2.1 Key Concepts
- **Command**: A callable unit in fastWorkflow, defined in a Python file (e.g., `add_user.py`) with a `Signature` class. Qualified name example: `ChatRoom/add_user`.
- **Parameter Metadata**: For each field in `Input` or `Output`:
  - **Name**: The field name (e.g., `user_name`).
  - **Type**: The Pydantic type annotation (e.g., `str`, `List[int]`; serialized as a string for matching).
  - **Description**: The `description` from `Field` (e.g., "Name of a person").
  - **Examples**: The `examples` from `Field` (e.g., `['John', 'Jane Doe']`).
  - **Pattern/Enum**: Optional constraints like regex patterns or enums.
- **Dependency Edge**: A directed link Y → X if at least one output parameter of X matches an input parameter of Y based on criteria in Section 3.
- **Graph**: A directed graph where nodes are command qualified names, and edges represent parameter dependencies (Y depends on X). Multi-edges are possible if multiple parameters match.

### 2.2 Input Data Sources
- **Command Directory**: Use `CommandDirectory` to access all command metadata and extract `Signature.Input` and `Signature.Output` information using the .
- **Routing Definition**: Use `RoutingDefinition` to resolve commands available in contexts, ensuring dependencies respect hierarchies.

## 3. Matching Criteria

Matching compares an output parameter from command X to an input parameter from command Y. Matches are scored and added as edges if they meet thresholds.

### 3.1 Exact Keyword Matching (Priority 1)
- **Criteria**:
  - **Name Match**: Exact string equality (case-insensitive) between X's output name and Y's input name.
  - **Type Match**: Exact match on serialized type (e.g., "str" matches "str"; "List[int]" matches "List[int]").
- **Score**: Binary (1.0 if both match, else 0.0).
- **Process**: For each pair (X.output_param, Y.input_param), check if score == 1.0. If yes, record for edge Y → X with metadata (e.g., {"matched_params": [("Y_input", "X_output")], "match_type": "exact"}).

### 3.2 Semantic Matching (Fallback if No Exact Match)
- **Criteria**: Compute similarity if exact match fails.
  - **Vectorization**: Embed a concatenated string of metadata: `"name: {name} | type: {type} | description: {description} | examples: {examples}"`.
    - Use a pre-trained embedding model (e.g., `all-MiniLM-L6-v2` from Sentence Transformers, integrated via a new utility in `utils/embeddings.py`).
  - **Similarity Score**: Cosine similarity between X's output embedding and Y's input embedding.
- **Threshold**: Configurable (default: 0.85). Record if similarity >= threshold.
- **Score Breakdown** (for diagnostics):
  - Name similarity: 40% weight.
  - Type similarity: 30% weight (fuzzy match, e.g., "string" ~ "str").
  - Description similarity: 20% weight.
  - Examples similarity: 10% weight.
- **Process**: Compute for non-exact pairs; record with metadata (e.g., {"matched_params": [...], "match_type": "semantic", "score": 0.92}).

### 3.3 Additional Rules
- **Self-Dependencies**: No edges from a command to itself.
- **Context Constraints**: Only match if X and Y are accessible in overlapping contexts (queried via `RoutingDefinition`).
- **Multi-Parameter Matches**: If multiple parameters match, aggregate scores (e.g., average) for edge weight.
- **Edge Weight**: Normalized score (0.0-1.0) indicating match strength.
- **Threshold Configuration**: CLI flags (e.g., `--exact-only`, `--semantic-threshold=0.8`).

#### 3.3.1 Elaboration on Context Constraints
Contexts in fastWorkflow are scopes tying commands to application objects (e.g., `User`, `PremiumUser`), defined by _commands/ directory structure and inherited via context_inheritance_model.json. 'Overlapping contexts' means X and Y share compatibility: direct (same context), inheritance (Y inherits X's context), or hierarchy (via parent-child relations).

**Why Needed**: Prevents invalid dependencies (e.g., suggesting an Admin-only X for a User-context Y), respects workflow structure, and ensures feasible suggestions.

**Example** (from messaging_app_3): Y = PremiumUser/send_priority_message overlaps with X = User/send_message (inheritance), but not with an unrelated Admin command.

**Implementation**: Before adding Y → X, query RoutingDefinition.get_contexts_for_command() for X and Y; add edge only if sets intersect or relate via inheritance/hierarchy.

## 4. Graph Construction Algorithm

### 4.1 Steps
1. **Extract Parameters**:
   - Scan all commands using `CommandDirectory` and `ast_class_extractor` (moved to `utils/ast_class_extractor.py` for shared use).
   - For each command, collect lists of input/output parameters as dicts: `{"name": str, "type": str, "description": str, "examples": list}`.

2. **Build Node List**: All qualified command names from `CommandDirectory`.

3. **Compute Matches**:
   - For every pair (X, Y) where X != Y:
     - For each output_param in X.outputs:
       - For each input_param in Y.inputs:
         - Compute exact match score.
         - If < 1.0, compute semantic score.
         - If score >= threshold, record match (for edge Y → X).
   - Optimize: Use embedding matrices for batch cosine similarity.

4. **Construct Graph**:
   - Use NetworkX (new dependency: `networkx`).
   - Add nodes for each command.
   - Add directed edges **Y → X** with attributes: `weight` (score), `matched_params` (list of tuples [(Y_input_name, X_output_name)]), `match_type` ("exact" or "semantic").

5. **Validate Graph**:
   - Detect cycles (error if found, as it implies circular dependencies).
   - Ensure no isolated nodes (warning if a command has no dependencies).

6. **Output Artifact**:
   - JSON: `{"nodes": [list], "edges": [{"from": "Y", "to": "X", "weight": float, "details": dict}]}`.
   - Save to `___command_info/parameter_dependency_graph.json`.

### 4.2 Example
Assume:
- Command X: `get_user` (Output: `{"user_id": {"type": "str", "desc": "User identifier"}}`)
- Command Y: `get_order` (Input: `{"user_id": {"type": "str", "desc": "ID of the user"}}`)

- Match on "user_id" → Edge **Y → X** (weight=1.0, matched_params=[("user_id", "user_id")]).

## 5. Implementation Details

All core graph functionality (generation, semantic search, querying) is consolidated into a single file: `utils/command_dependency_graph.py`. Other components (e.g., build, validation) import from this file.

### 5.1 Integration Points
- **Build Phase**: Extend `build/__main__.py` to call `generate_dependency_graph(workflow_path)` from `utils/command_dependency_graph.py` automatically on every build (no separate flag).
- **Semantic Search**: Implement in `command_dependency_graph.py` using embeddings (e.g., Sentence Transformers).
- **Runtime Query Function**: In `command_dependency_graph.py`:
  ```python
  def get_dependency_suggestions(
      graph_path: str,  # Path to ___command_info/parameter_dependency_graph.json
      y_qualified_name: str,
      missing_input_param: str,
      min_weight: float = 0.7,  # Configurable threshold
      max_depth: int = 3  # Prevent deep recursion
  ) -> List[Dict]:  # List of dependency plans
      """
      Recursively resolves dependencies, returning a list of plans (each a dict representing a tree or chain)
      for resolving the missing param, including order and nested dependencies.
      """
      import networkx as nx
      G = nx.read_gpickle(graph_path)

      def recurse(node: str, param: str, path: List[str], depth: int) -> List[Dict]:
          if depth > max_depth:
              return []
          plans = []
          for neighbor in G.neighbors(node):
              edge_data = G.get_edge_data(node, neighbor)
              if edge_data['weight'] >= min_weight and any(p[0] == param for p in edge_data['matched_params']):
                  new_path = path + [neighbor]
                  sub_plans = recurse(neighbor, param, new_path, depth + 1)  # Recurse for neighbor's deps
                  plans.append({
                      'command': neighbor,
                      'sub_plans': sub_plans
                  })
          return plans

      # Start recursion from Y
      dependency_plans = recurse(y_qualified_name, missing_input_param, [], 0)

      # Sort plans by some criteria, e.g., shortest path or highest weight
      dependency_plans.sort(key=lambda p: len(p['sub_plans']))  # Example: prefer shallower trees
      return dependency_plans
  ```
  - Integrate into error handling: When validation fails for missing Y.input_param, call this and suggest: "Suggested resolution plan: Call X (then Z if needed) to get the value."

#### 5.1.1 Validation Message Refinement
Add `refine_validation_message(missing_fields: List[str], y_qualified_name: str, graph_path: str) -> str` in `utils/signatures.py` under `InputForParamExtraction`. Called by `validate_parameters()` to augment error messages.

**Logic**:
- For each missing field, call `get_dependency_suggestions()` to get plans.
- If plans exist, append suggestions (e.g., "To get {field}, try this plan: Call X, then Y.").
- If no plan, leave the original message for that field (e.g., "Missing required field: {field}").
- Combine into a single refined message.

**Example Augmented Message**: "Missing 'user_id'. Suggested plan: Call 'get_user_by_email' (requires 'email'). Original error for 'order_id': Missing required field."

### 5.2 Performance Considerations
- **Complexity**: O(C^2 * P^2) where C=commands, P=params per command. Optimize with vectorized embeddings.
- **Query Time**: O(E * D) where E is edges from Y, D is max_depth (bound recursion).
- **Caching**: Cache embeddings per parameter in `command_dependency_graph.py`.

### 5.3 Error Handling
- Invalid types/descriptions: Skip and log.
- No matches: Empty graph for that pair.
- If no suggestions: Return empty list (log "No dependencies found for Y.input_param").
- Recursion depth exceeded: Return partial plans with warning.

## 6. Usage and Extensions

### 6.1 Runtime Usage
- In validation error handling (e.g., `validate_parameters` in `utils/signatures.py`): If missing param, call `get_dependency_suggestions` and include plans in error message/suggestions.
- When Y lacks input, query graph for predecessors of Y and suggest/calls (future feature).

### 6.2 Extensions
- **Dynamic Dependencies**: Runtime graph updates.
- **Multi-Workflow**: Cross-workflow edges.
- **User Overrides**: Manual edge additions via JSON.
- **Graph Visualization**: Export to GraphML for tools like Gephi.

This spec provides a foundation for implementing parameter dependency graphs, enhancing fastWorkflow's reliability for complex agents. 