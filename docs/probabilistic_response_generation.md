# Probabilistic Response Generation in fastWorkflow

## 1. Overview

### 1.1 Purpose
This specification outlines an optional extension to fastWorkflow's command system, enabling probabilistic (AI-generated) response generation for commands. Currently, ResponseGenerator implementations are deterministic, manually coded wrappers around application logic. This feature allows developers to stub commands with a docstring and a simple \"pass\" statement, triggering GenAI to auto-implement probabilistic logic during the build phase. It introduces post-conditions on Signature.Output fields, optional scoring functions (code-based or natural language for LLM evaluation), and DSPy optimization (defaulting to MIPROv2) during a new training phase.

The goal is to support adaptive, non-deterministic responses for complex or creative tasks, while maintaining backward compatibility and optionality. Probabilistic generation is only activated for explicitly stubbed commands, ensuring no impact on existing deterministic workflows.

### 1.2 Scope
- **In Scope**:
  - Extensions to Signature.Output for optional post-conditions.
  - Build-time GenAI implementation of _process_command() for stubbed ResponseGenerators.
  - Optional scoring functions and DSPy optimization per command.
  - New train_response_generation package for optimizing probabilistic commands using synthetic data.
  - Human-in-the-loop scoring in a dedicated training mode.
  - Integration with existing build and train pipelines.
- **Out of Scope**:
  - Runtime probabilistic execution for non-stubbed commands.
  - Cross-command optimization or workflow-level probabilistic planning (future extension).
  - Advanced DSPy customizations beyond standard algorithms.

### 1.3 Key Benefits
- **Rapid Prototyping**: Auto-generate probabilistic implementations for stubs, accelerating development for uncertain or creative commands.
- **Adaptive Responses**: Enable outputs that vary based on context, with built-in quality checks via post-conditions and scoring.
- **Optimization Loop**: Use synthetic data and human feedback to improve response quality without manual coding.
- **Flexibility**: Fully optional; preserves deterministic simplicity for most use cases.
- **Debugging Aids**: Post-conditions provide validation even for deterministic commands.

### 1.4 Assumptions and Dependencies
- **Dependencies**:
  - DSPy library for optimization (already integrated in fastWorkflow).
  - GenAI capabilities in build tool (extend from existing post-processing specs like build_postprocessing_using_dspy.md).
  - Existing synthetic data generation (e.g., generate_diverse_utterances).
  - Pydantic for Signature extensions.
- **Assumptions**:
  - Users provide a docstring in _process_command() stubs to guide GenAI.
  - Training mode is run separately (e.g., via CLI flag) to avoid disrupting normal workflows.
  - Environment variables (e.g., LLM keys) are set for GenAI and DSPy.
  - Build-time validation ensures generated code is safe and functional.

## 2. Architecture

### 2.1 High-Level Components
1. **Extended Signature Class**: Adds optional post-conditions to Output fields.
2. **Stub Detection in Build Tool**: Scans for _process_command() stubs and replaces with GenAI-generated DSPy code.
3. **Scoring Mechanism**: Per-command function (code or NL) to evaluate outputs against post-conditions.
4. **DSPy Integration**: Optional optimizer (e.g., MIPROv2) applied during training.
5. **train_response_generation Package**: Generates synthetic input-output pairs, scores them (with human-in-loop option), and optimizes DSPy programs.
6. **Runtime Wrapper**: For probabilistic commands, wraps generated logic with scoring and retries.

### 2.2 Execution Flow
- **Build Phase**:
  1. Scan commands for stubs (docstring + \"pass\") in ResponseGenerator._process_command().
  2. Use GenAI (via DSPy signature) to generate DSPy-based implementation, incorporating post-conditions.
  3. Validate generated code (compile, run basic tests against synthetics).
  4. Replace stub if valid; log and skip if not.

- **Train Phase (train_response_generation)**:
  1. For probabilistic commands, generate synthetic utterances/inputs (extend generate_diverse_utterances).
  2. Run generated _process_command() to produce outputs.
  3. Score outputs (code/LLM or human-in-loop in training mode).
  4. Apply DSPy optimization (e.g., MIPROv2) using scores as metrics.
  5. Save optimized DSPy program for runtime use.

- **Runtime**:
  1. For probabilistic commands, load optimized DSPy program.
  2. Generate response, check post-conditions, score, and retry if needed.
  3. Fallback to deterministic if probabilistic fails (configurable).

### 2.3 DSPy Components
- **Generation Signature**: For build-time code gen (e.g., \"Generate DSPy code for _process_command() that satisfies [post-conditions] and uses [scoring].\").
- **Scoring Signature**: If NL-based, convert to DSPy metric (e.g., \"Score output on scale 0-10 based on [instructions].\").
- **Optimization**: Use MIPROv2 by default for compiling DSPy programs with synthetic examples and scores.

## 3. Detailed Features

### 3.1 Extended Signature.Output with Post-Conditions
- **Definition**: Add optional `post_conditions` to Output fields (e.g., via json_schema_extra: {\"post_conditions\": \"must be > input.value\"}).
- **Usage**: Checked post-generation; if failed, trigger retry or error.
- **Implementation**: Extend Pydantic validation in utils/signatures.py to evaluate post-conditions after _process_command().

### 3.2 Optional Scoring Function
- **Types**: Code (Python function) or NL (instructions for LLM-based scoring).
- **Definition**: Add to Signature (e.g., scoring_func: Callable or str).
- **Human-in-the-Loop**: In training mode, prompt users for scores on generated outputs; store for DSPy metrics.
- **Runtime**: Score outputs; if below threshold, regenerate (configurable max_retries).

### 3.3 Optional DSPy Optimization
- **Specification**: Add to Signature (e.g., optimizer: str = \"MIPROv2\").
- **Process**: During training, compile DSPy program for _process_command() using synthetics and scores.

### 3.4 Build Integration
- **Stub Detection**: In command_stub_generator.py, check for \"pass\" in _process_command().
- **GenAI Replacement**: Use DSPy to generate code; insert into file.
- **Validation**: Auto-generate/run tests (e.g., check post-conditions on sample inputs).

### 3.5 train_response_generation Package
- **Inputs**: Synthetic utterances from generate_utterances; expected outputs from post-conditions.
- **Process**: Generate outputs, score (auto or human), optimize with DSPy.
- **Training Mode**: CLI flag (--human-scoring) for interactive feedback.
- **Output**: Optimized DSPy programs saved per command (e.g., in ___command_info/).

## 4. Implementation Details
- **Signature Changes**: Add fields like post_conditions (dict), scoring_func (Union[Callable, str]), optimizer (str).
- **ResponseGenerator**: For probabilistic, wrap generated DSPy code with scoring/retries.
- **Build Tool**: Extend __main__.py with post-stub phase for GenAI.
- **New Package**: train_response_generation.py with synthetic gen, scoring loop, DSPy compiler.
- **Error Handling**: Build failures revert stubs; runtime failures fallback to deterministic.
- **Performance**: Cache DSPy programs; limit retries.

## 5. Testing and Validation
- **Unit Tests**: Signature extensions, scoring functions, GenAI output validation.
- **Integration Tests**: End-to-end build/train/run for stubbed commands.
- **Edge Cases**: Failed generations, low scores, human-loop interruptions. 