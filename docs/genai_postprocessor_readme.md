# GenAI Post-Processor for FastWorkflow Build Tool

## Overview

The GenAI Post-Processor is an integrated component of the FastWorkflow build tool that automatically enhances generated command files using AI-powered content generation. It leverages DSPy (Declarative Self-improving Language Programs) to generate high-quality metadata, natural language utterances, and documentation for your workflow commands. This functionality is always enabled and uses the FastWorkflow environment configuration.

## Features

### 1. Field Metadata Enhancement
- **Descriptions**: Generates clear, concise descriptions for input/output fields
- **Examples**: Provides 2-3 realistic example values for each field
- **Patterns**: Generates regex validation patterns for string fields (e.g., `^[A-Z][a-z]+$` for capitalized words, `^\d{3}-\d{3}-\d{4}$` for phone numbers)

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

The GenAI post-processor is included with FastWorkflow. DSPy is already included in the FastWorkflow dependencies.

## Configuration

The post-processor uses the FastWorkflow environment configuration:

### Environment Variables

Set these in your `.env` file or environment:

```bash
# Model configuration (supports LiteLLM model strings)
LLM_COMMAND_METADATA_GEN=mistral/mistral-small-latest
# or
LLM_COMMAND_METADATA_GEN=bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0

# API key for the model provider
LITELLM_API_KEY_COMMANDMETADATA_GEN=your_api_key_here
```

### Supported Models

Through LiteLLM, the post-processor supports:
- **OpenAI**: gpt-4, gpt-3.5-turbo, etc.
- **Anthropic**: claude-3-opus, claude-3-sonnet, etc.
- **AWS Bedrock**: bedrock/anthropic.claude-*, bedrock/meta.llama*, etc.
- **Mistral**: mistral-small-latest, mistral-large-latest, etc.
- **Google**: gemini-pro, gemini-1.5-pro, etc.
- **Local models**: Via Ollama or other local providers
- **100+ other providers**: See [LiteLLM documentation](https://docs.litellm.ai/docs/providers)

## Usage

### Basic Usage

The post-processor runs automatically when you build your workflow:

```bash
fastworkflow build --app-dir ./my_app --workflow-folderpath ./my_workflow
```

The post-processing step is always enabled and will enhance all generated command files.

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
   - Automatically configures DSPy using environment variables
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
            pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"  # UUID pattern
        )
        action: str = Field(
            description="The action to perform on the user account",
            examples=["create", "update", "delete"],
            pattern="^(create|update|delete)$"  # Enum-like pattern
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
- **Missing Configuration**: Uses default model if environment variables are not set
- **No API Key**: Continues with warning if API key is missing

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

### Using Different AI Providers

The post-processor uses LiteLLM through DSPy, which supports 100+ providers out of the box. Simply set the appropriate model string in your environment:

```bash
# OpenAI
LLM_COMMAND_METADATA_GEN=gpt-4

# Anthropic via Bedrock
LLM_COMMAND_METADATA_GEN=bedrock/anthropic.claude-3-opus-20240229-v1:0

# Google Gemini
LLM_COMMAND_METADATA_GEN=gemini/gemini-1.5-pro

# Local Ollama
LLM_COMMAND_METADATA_GEN=ollama/llama3
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
   - Solution: Set `LITELLM_API_KEY_COMMANDMETADATA_GEN` environment variable

2. **Model Not Found**
   - Solution: Ensure `LLM_COMMAND_METADATA_GEN` contains a valid LiteLLM model string
   - Check [LiteLLM documentation](https://docs.litellm.ai/docs/providers) for correct format

3. **Rate Limiting**
   - Solution: Use a different model with higher rate limits
   - Consider implementing retry logic with exponential backoff

4. **Large Workflows**
   - Solution: Use a faster model like `mistral/mistral-small-latest` for development

## Best Practices

1. **Development Workflow**
   - Use faster models during development (e.g., `mistral/mistral-small-latest`)
   - Use more powerful models for production (e.g., `gpt-4`, `claude-3-opus`)
   - Review generated content for accuracy

2. **Model Selection**
   - Use Claude or GPT-4 for highest quality
   - Use Mistral or GPT-3.5-turbo for faster, cheaper processing
   - Consider local models via Ollama for sensitive data

3. **Content Validation**
   - Always review generated utterances for naturalness
   - Verify regex patterns are valid and match expected input formats
   - Test patterns to ensure they don't reject valid inputs
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
