# GenAI Post-Processor for FastWorkflow Build Tool

## Overview

The GenAI Post-Processor is an advanced component of the FastWorkflow build tool that enhances generated command files using AI-powered content generation. It leverages DSPy (Declarative Self-improving Language Programs) to generate high-quality metadata, natural language utterances, and documentation for your workflow commands.

## Features

### 1. Field Metadata Enhancement
- **Descriptions**: Generates clear, concise descriptions for input/output fields
- **Examples**: Provides 2-3 realistic example values for each field
- **Constraints**: Documents validation rules and patterns (e.g., "must be positive integer", "valid email format")

### 2. Natural Language Utterance Generation
- Creates minimal sets of natural language commands
- Covers all parameter combinations (none, single, multiple)
- Varies parameter values across utterances for diversity
- Ensures natural, conversational phrasing

### 3. Dynamic Docstring Generation
- **Command Signatures**: Comprehensive Google-style docstrings for command classes
- **Context Handlers**: Aggregated docstrings summarizing context capabilities
- **Workflow Description**: High-level overview of the entire workflow system

### 4. AST-Based File Updates
- Preserves existing code structure and logic
- Updates files in-place using Python AST manipulation
- Maintains code formatting and organization
- Handles errors gracefully without breaking the build

## Installation

The GenAI post-processor is included with FastWorkflow. Ensure you have DSPy installed:

```bash
pip install dspy-ai
```

## Usage

### Basic Usage

Run the build tool with GenAI post-processing enabled (default):

```bash
fastworkflow build --app-dir ./my_app --workflow-folderpath ./my_workflow
```

### Configuration Options

#### Disable Post-Processing
```bash
fastworkflow build --app-dir ./my_app --workflow-folderpath ./my_workflow --no-genai
```

#### Specify AI Model
```bash
fastworkflow build --app-dir ./my_app --workflow-folderpath ./my_workflow --genai-model gpt-3.5-turbo
```

#### Provide API Key
```bash
fastworkflow build --app-dir ./my_app --workflow-folderpath ./my_workflow --genai-api-key YOUR_API_KEY
```

Or set via environment variable:
```bash
export OPENAI_API_KEY=YOUR_API_KEY
fastworkflow build --app-dir ./my_app --workflow-folderpath ./my_workflow
```

### Supported Models

- **OpenAI**: gpt-4, gpt-3.5-turbo
- **Other providers**: Can be extended to support Anthropic, Cohere, local LLMs, etc.

## Architecture

### Components

1. **DSPy Signatures**: Define input/output schemas for AI tasks
   - `FieldMetadataSignature`: For field descriptions and examples
   - `UtteranceGeneratorSignature`: For natural language commands
   - `SignatureDocstringSignature`: For command documentation
   - `ContextDocstringSignature`: For context handler documentation
   - `WorkflowDescriptionSignature`: For workflow overview

2. **DSPy Modules**: Implement AI-powered generation logic
   - `FieldMetadataGenerator`: Generates field metadata using Chain of Thought
   - `UtteranceGenerator`: Creates natural language utterances
   - `DocstringGenerator`: Produces comprehensive documentation
   - `WorkflowDescriptionGenerator`: Creates workflow descriptions

3. **GenAIPostProcessor**: Main orchestrator class
   - Manages DSPy configuration
   - Processes command files
   - Updates AST structures
   - Handles error recovery

### Processing Flow

1. **Initialization**: Configure DSPy with specified model and API key
2. **Command File Processing**:
   - Parse existing command files using AST
   - Extract input/output field information
   - Generate enhanced metadata using DSPy
   - Update AST with new content
   - Write updated files back to disk
3. **Context Handler Processing**:
   - Aggregate command docstrings
   - Generate context-level documentation
   - Update handler files with docstrings
4. **Workflow Description**:
   - Collect all contexts and commands
   - Generate comprehensive workflow overview
   - Write to `workflow_description.txt`

## Example Output

### Before Post-Processing
```python
class Signature:
    class Input(BaseModel):
        user_id: str
        action: str
    
    plain_utterances = ["execute action"]
```

### After Post-Processing
```python
class Signature:
    """Process user actions in the system.
    
    Args:
        user_id: Unique identifier for the user. Must be a valid UUID.
        action: The action to perform. Valid actions include 'create', 'update', 'delete'.
    
    Returns:
        success: Whether the action was successful
        message: Detailed result message
    
    Examples:
        >>> process_user_action(user_id="123e4567-e89b-12d3-a456-426614174000", action="create")
        Output(success=True, message="User created successfully")
    """
    
    class Input(BaseModel):
        user_id: str = Field(
            description="Unique identifier for the user",
            examples=["123e4567-e89b-12d3-a456-426614174000", "user_12345"],
            json_schema_extra={"constraints": "must be valid UUID format"}
        )
        action: str = Field(
            description="The action to perform on the user account",
            examples=["create", "update", "delete"],
            json_schema_extra={"constraints": "must be one of: create, update, delete"}
        )
    
    plain_utterances = [
        "perform action",
        "execute create for user 123",
        "update user abc with action delete",
        "process user 456 action update"
    ]
```

## Error Handling

The post-processor is designed to be fault-tolerant:

- **API Failures**: Falls back to default values if AI generation fails
- **Invalid Files**: Skips files that can't be parsed
- **Missing Dependencies**: Warns but continues if DSPy is not available
- **No API Key**: Skips post-processing with warning

The build process never fails due to post-processing errors.

## Testing

### Unit Tests
```bash
python -m pytest tests/test_genai_postprocessor.py
```

### Integration Tests
```bash
python -m pytest tests/test_genai_postprocessor_integration.py
```

### Test Coverage
- DSPy signature definitions
- Module implementations
- AST manipulation
- Error recovery
- CLI integration
- Full workflow processing

## Performance Considerations

- **Caching**: Results are cached to avoid redundant API calls
- **Batch Processing**: Can be extended to batch multiple fields/commands
- **Parallel Processing**: Future enhancement for processing multiple files concurrently
- **Model Selection**: Use faster models (e.g., gpt-3.5-turbo) for development

## Extending the Post-Processor

### Adding New AI Providers

```python
def _initialize_dspy(self):
    if self.model.startswith("claude"):
        lm = dspy.Claude(model=self.model, api_key=self.api_key)
    elif self.model.startswith("llama"):
        lm = dspy.LocalLLM(model_path=self.model)
    # ... other providers
```

### Custom Generation Strategies

```python
class CustomSignature(dspy.Signature):
    """Your custom generation task."""
    # Define input/output fields
    
class CustomGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CustomSignature)
```

### Optimization with DSPy

DSPy supports automatic prompt optimization:

```python
# Compile with examples for better performance
examples = [...]
teleprompter = BootstrapFewShot(metric=my_metric)
compiled_generator = teleprompter.compile(generator, examples)
```

## Troubleshooting

### Common Issues

1. **No API Key Error**
   - Solution: Set `OPENAI_API_KEY` environment variable or use `--genai-api-key` flag

2. **Import Error for DSPy**
   - Solution: Install DSPy with `pip install dspy-ai`

3. **Rate Limiting**
   - Solution: Use `--genai-model gpt-3.5-turbo` for higher rate limits
   - Consider implementing retry logic with exponential backoff

4. **Large Workflows**
   - Solution: Process in batches or use `--no-genai` for initial development

## Best Practices

1. **Development Workflow**
   - Use `--no-genai` during rapid iteration
   - Enable post-processing for production builds
   - Review generated content for accuracy

2. **Model Selection**
   - Use GPT-4 for highest quality
   - Use GPT-3.5-turbo for faster, cheaper processing
   - Consider local models for sensitive data

3. **Content Validation**
   - Always review generated utterances for naturalness
   - Verify field constraints match actual validation
   - Ensure examples are realistic and valid

## Future Enhancements

- [ ] Support for more AI providers (Anthropic, Cohere, Gemini)
- [ ] Parallel processing of multiple files
- [ ] Incremental updates (only process changed files)
- [ ] Custom prompt templates
- [ ] Fine-tuned models for workflow-specific generation
- [ ] Integration with workflow testing to validate generated content

## Contributing

Contributions are welcome! Please see the main FastWorkflow contributing guidelines.

## License

This component is part of FastWorkflow and is licensed under the Apache-2.0 License.