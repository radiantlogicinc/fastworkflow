"# Specification for GenAI Post-Processing in FastWorkflow Build Tool

## Overview

This specification outlines a post-processing component for the FastWorkflow build tool to address identified flaws in generated command files. The component will use DSPy (a framework for programming with foundation models) to generate non-deterministic, high-quality content such as field descriptions, examples, constraints, utterances, and docstrings. This will enhance the usability, accuracy, and completeness of generated command files, context handlers, and overall workflow documentation.

The post-processor runs after the deterministic build phase (which generates skeleton command files, context models, etc.) and updates files in place with AI-generated improvements. It will also generate new artifacts like workflow_description.txt.

### Key Goals
- Improve input/output field metadata with rich descriptions, examples, and constraints.
- Generate a minimal set of natural utterances based on input parameters, covering every combination of parameters (including none), with varied parameter values across utterances.
- Add dynamic docstrings to Command Signature classes.
- Create docstrings for _<Context>.py files based on aggregated command docstrings.
- Produce a workflow_description.txt summarizing the entire workflow.

### Assumptions and Dependencies
- **DSPy Integration**: The post-processor assumes DSPy is installed and configured with access to a foundation model (e.g., via OpenAI, Grok, or local LLMs).
- **Model Selection**: Use a capable model like GPT-4 or Grok for generation. DSPy will handle optimization and compilation.
- **Input Data**: Relies on AST-parsed class/method info from the build tool (e.g., ClassInfo, MethodInfo objects).
- **Execution Timing**: Runs as a separate phase after `generate_command_files` in `command_file_generator.py`.
- **Error Handling**: If GenAI fails (e.g., API errors), fall back to deterministic defaults and log warnings.

## Architecture

### High-Level Flow
1. **Deterministic Build Phase**: Run the existing build tool to generate skeleton command files, context models, and directory structure.
2. **Post-Processing Phase**:
   - Load generated files and context model.
   - For each command file (in contexts and global):
     - Use DSPy to generate improved input/output fields, utterances, and docstrings.
     - Update the file in place.
   - For each context's _<Context>.py file:
     - Aggregate command docstrings and generate a context-level docstring.
   - Generate workflow_description.txt by summarizing all contexts and global commands.
3. **Output**: Updated command files, new docstrings, and workflow description.

### DSPy Components
DSPy programs will be defined as signatures (prompt templates) and modules (chains or programs). Each will be compiled with examples for optimization.

#### 1. Field Metadata Generator
- **Purpose**: Generate descriptions, examples, and constraints for input/output fields.
- **DSPy Signature**:
  ```
  class FieldMetadata(dspy.Signature):
      \"\"\"Generate metadata for a field.\"\"\"
      field_name: str
      field_type: str
      method_docstring: str
      method_name: str
      context_name: str
      is_input: bool

      description: str = dspy.OutputField(desc=\"Clear, concise description of the field.\")
      examples: List[str] = dspy.OutputField(desc=\"2-3 example values.\")
      constraints: str = dspy.OutputField(desc=\"Any patterns or constraints (e.g., 'must be positive integer').\")
  ```
- **Module**: ChainOfThought(FieldMetadata) – Use reasoning to generate metadata.
- **Usage**: For each field in Input/Output, call the module and update Field(description=..., examples=..., json_schema_extra={\"examples\": examples}).

#### 2. Utterance Generator
- **Purpose**: Generate a minimal set of natural utterances based on input parameters, covering every combination of parameters (including none), with varied parameter values across utterances.
- **DSPy Signature**:
  ```
  class UtteranceGenerator(dspy.Signature):
      \"\"\"Generate minimal natural language utterances for a command.\"\"\"
      command_name: str
      command_docstring: str
      input_fields: List[Dict[str, str]]  # name, type, desc

      utterances: List[str] = dspy.OutputField(desc=\"Minimal list of utterances covering all parameter combinations, with varied values.\")
  ```
- **Module**: ChainOfThought(UtteranceGenerator).
- **Usage**: Replace hardcoded plain_utterances with generated list.

#### 3. Docstring Generator
- **Purpose**: Generate docstrings for Signature class and _<Context>.py.
- **DSPy Signature** (for Signature):
  ```
  class SignatureDocstring(dspy.Signature):
      \"\"\"Generate docstring for Command Signature.\"\"\"
      command_name: str
      input_fields: List[Dict[str, str]]
      output_fields: List[Dict[str, str]]
      context_name: str

      docstring: str = dspy.OutputField(desc=\"Comprehensive docstring in Google style.\")
  ```
- **For _<Context>.py**:
  ```
  class ContextDocstring(dspy.Signature):
      \"\"\"Generate docstring for context handler.\"\"\"
      context_name: str
      commands: List[Dict[str, str]]  # name, docstring

      docstring: str = dspy.OutputField(desc=\"Aggregated docstring summarizing context and commands.\")
  ```
- **Module**: ChainOfThought for both.

#### 4. Workflow Description Generator
- **Purpose**: Create workflow_description.txt summarizing everything.
- **DSPy Signature**:
  ```
  class WorkflowDescription(dspy.Signature):
      \"\"\"Generate overall workflow description.\"\"\"
      contexts: List[Dict[str, Any]]  # context_name, docstring, commands
      global_commands: List[Dict[str, str]]

      description: str = dspy.OutputField(desc=\"High-level workflow overview.\")
  ```
- **Module**: ChainOfThought.
- **Output**: Write to workflow_description.txt in workflow root.

### Execution Workflow
- **Entry Point**: Add to `__main__.py` after generation: `run_genai_postprocessor(args)`.
- **Steps**:
  1. Load context model and scan generated files.
  2. Initialize DSPy program (compile with few-shot examples).
  3. For each command file: Parse AST, generate metadata/utterances/docstrings, rewrite file.
  4. For each context: Aggregate, generate _<Context>.py docstring.
  5. Generate workflow_description.txt.
- **Optimization**: Use DSPy's teleprompters for compiling programs with examples.
- **Caching**: Cache GenAI outputs to avoid repeated calls.

## Implementation Details
- **File Updates**: Use AST manipulation to insert generated content without overwriting logic.
- **Error Recovery**: If DSPy fails, log and use defaults.
- **Performance**: Batch requests if possible; limit to one call per item.
- **Testing**: Unit tests for DSPy signatures; integration tests for full post-processing. 