# GenAI Post-Processor for FastWorkflow Build Tool

## Overview

The GenAI Post-Processor is an integrated component of the FastWorkflow build tool that automatically enhances generated command files using AI-powered content generation. It leverages DSPy (Declarative Self-improving Language Programs) to generate high-quality metadata, natural language utterances, and documentation for your workflow commands.

> **🚀 LibCST Implementation**: The GenAI postprocessor uses LibCST (Concrete Syntax Tree) for targeted, source-preserving updates that make laser-focused changes without modifying existing content. All formatting, comments, and manual edits are preserved. The old ast.unparse() implementation has been completely replaced.

## Features

### 1. Targeted Field Metadata Enhancement
- **Descriptions**: Adds clear, concise descriptions ONLY for fields missing them
- **Examples**: Provides 2-3 realistic example values ONLY where absent
- **Patterns**: Generates regex validation patterns for string fields when appropriate
- **Annotated Support**: Correctly handles both regular fields and `Annotated[type, Field(...)]` syntax
- **Preservation**: Never overwrites existing field metadata

### 2. Intelligent Utterance Management
- **Appends Only**: New utterances are added to existing lists, never replacing them
- **No Duplicates**: Checks for existing utterances before adding
- **Natural Language**: Creates conversational commands covering various parameter combinations
- **Preserves Manual Edits**: Custom utterances added by developers are always kept

### 3. Smart Docstring Generation
- **Only If Missing**: Signature docstrings are added only when absent or empty
- **Google Style**: Generates comprehensive Google-style docstrings
- **Context Handlers**: Creates aggregated docstrings for context capabilities
- **Workflow Description**: Generates high-level overview in workflow_description.txt

### 4. LibCST-Based Surgical Updates
- **Source Preservation**: Maintains exact formatting, comments, and code style
- **Minimal Changes**: Only modifies what needs to be changed
- **Idempotent**: Running multiple times won't duplicate or alter existing content
- **Error Resilient**: Handles errors gracefully without breaking the build

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

### Core Components

1. **LibCST Transformers** (`libcst_transformers.py`):
   - `SignatureDocstringUpdater`: Adds docstrings only if missing or empty
   - `FieldMetadataUpdater`: Adds field metadata only where absent, handles Annotated fields
   - `UtteranceAppender`: Appends new utterances without removing existing ones
   - `StateExtractor`: Analyzes current file state to determine what needs updating

2. **DSPy Signatures**: Define input/output schemas for AI tasks
   - `FieldMetadataSignature`: For field descriptions and examples
   - `UtteranceGeneratorSignature`: For natural language commands
   - `SignatureDocstringSignature`: For command documentation
   - `ContextDocstringSignature`: For context handler documentation
   - `WorkflowDescriptionSignature`: For workflow overview

3. **DSPy Modules**: Implement AI-powered generation logic
   - `FieldMetadataGenerator`: Generates field metadata using Chain of Thought
   - `UtteranceGenerator`: Creates natural language utterances
   - `DocstringGenerator`: Produces comprehensive documentation
   - `WorkflowDescriptionGenerator`: Creates workflow descriptions

4. **GenAIPostProcessor**: Main orchestrator class
   - Configures DSPy using environment variables
   - Coordinates LibCST transformers for targeted updates
   - Provides detailed statistics on changes made
   - Handles error recovery gracefully

### Processing Flow

1. **Initialization**: Configure DSPy with specified model and API key
2. **State Analysis**:
   - Parse files using LibCST to preserve formatting
   - Extract current state using `StateExtractor`
   - Identify what content is missing (docstrings, metadata, utterances)
3. **Targeted Enhancement**:
   - Generate ONLY missing content using DSPy modules
   - Apply surgical updates using LibCST transformers
   - Skip files that don't need changes
4. **Context Handler Processing**:
   - Aggregate command docstrings
   - Generate context-level documentation
   - Update handler files with docstrings (only if missing)
5. **Workflow Description**:
   - Collect all contexts and commands
   - Generate comprehensive workflow overview
   - Write to `workflow_description.txt`
6. **Statistics Reporting**:
   - Report number of files processed
   - Show counts of docstrings, fields, and utterances added
   - Indicate files skipped (no changes needed)

## Example Output

### Before Post-Processing
```python
class Signature:
    class Input(BaseModel):
        user_id: str
        action: str
    
    plain_utterances = ["execute action"]
```

### After Post-Processing (LibCST)
```python
class Signature:
    """Process user actions in the system."""  # Added only if missing
    
    class Input(BaseModel):
        user_id: str = Field(  # Field() added only if missing
            description="Unique identifier for the user",
            examples=["123e4567-e89b-12d3-a456-426614174000", "user_12345"],
            pattern="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        action: str = Field(
            description="The action to perform on the user account",
            examples=["create", "update", "delete"],
            pattern="^(create|update|delete)$"
        )
    
    plain_utterances = [
        "execute action",  # Original preserved
        "execute create for user 123",  # New utterances appended
        "update user abc with action delete",
        "process user 456 action update"
    ]
```

### Special Case: Annotated Fields
```python
# Input with Annotated syntax
class Input(BaseModel):
    order_id: Annotated[
        str,
        Field(description="Order ID", examples=["#123"])
    ]  # No duplicate Field() assignment added
    
    name: str  # Regular field - gets Field() added

# After processing
class Input(BaseModel):
    order_id: Annotated[
        str,
        Field(description="Order ID", examples=["#123"])
    ]  # Correctly preserved, no duplication
    
    name: str = Field(
        description="Customer name",
        examples=["John Doe", "Jane Smith"]
    )  # Field() added only where needed
```

## Error Handling

The post-processor is designed to be fault-tolerant:

- **API Failures**: Falls back gracefully if AI generation fails
- **Invalid Files**: Skips files that can't be parsed with LibCST
- **Missing Configuration**: Uses default model if environment variables are not set
- **No API Key**: Continues with warning if API key is missing
- **Malformed AST**: Logs error and skips file without breaking build
- **Annotated Fields**: Correctly handles both regular and Annotated field syntax

The build process never fails due to post-processing errors. Files are only modified if updates can be safely applied.

## Testing

### Unit Tests
```bash
python -m pytest tests/test_genai_postprocessor.py
python -m pytest tests/test_libcst_transformers.py
```

### Integration Tests
```bash
python -m pytest tests/test_genai_postprocessor_integration.py
python -m pytest tests/test_libcst_integration.py
```

### Test Coverage
- LibCST transformer classes (14 test cases)
- DSPy signature definitions
- Module implementations
- Preservation of existing content
- Annotated field handling
- Idempotency verification
- Error recovery
- CLI integration
- Full workflow processing

## Performance Considerations

- **LibCST Processing**: Slightly slower than ast.unparse (10-20% overhead) but ensures correctness
- **Targeted Updates**: Only processes files that need changes, reducing overall time
- **Caching**: Results are cached to avoid redundant API calls
- **Batch Processing**: Can be extended to batch multiple fields/commands
- **Parallel Processing**: Future enhancement for processing multiple files concurrently
- **Model Selection**: Use faster models (e.g., mistral-small-latest) for development, powerful models (gpt-4, claude-3) for production

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
   - Run `fastworkflow refine` after initial build to enhance metadata
   - Manual edits are always preserved - feel free to customize
   - Re-running is safe - the tool is idempotent
   - Use faster models during development (e.g., `mistral/mistral-small-latest`)
   - Use more powerful models for production (e.g., `gpt-4`, `claude-3-opus`)

2. **Manual Customization**
   - Add custom utterances directly - they'll be preserved
   - Edit field descriptions/examples - they won't be overwritten
   - Add custom docstrings - they'll be kept intact
   - The postprocessor only fills in gaps, never replaces your work

3. **Model Selection**
   - Use Claude or GPT-4 for highest quality
   - Use Mistral or GPT-3.5-turbo for faster, cheaper processing
   - Consider local models via Ollama for sensitive data

4. **Content Validation**
   - Review generated utterances for naturalness
   - Verify regex patterns are valid and match expected input formats
   - Test patterns to ensure they don't reject valid inputs
   - Ensure examples are realistic and valid

## Implementation Details

### LibCST vs AST.unparse Comparison

| Feature | Old (ast.unparse) | New (LibCST) |
|---------|------------------|-------------|
| Formatting Preservation | ❌ Lost | ✅ Preserved |
| Comments | ❌ Removed | ✅ Kept |
| Existing Utterances | ❌ Replaced | ✅ Appended |
| Field Metadata | ❌ Overwritten | ✅ Merged |
| Import Order | ❌ Changed | ✅ Maintained |
| Annotated Fields | ❌ Duplicated | ✅ Handled Correctly |
| Idempotency | ❌ No | ✅ Yes |

### Key Implementation Files

- `fastworkflow/build/genai_postprocessor.py` - Main orchestrator
- `fastworkflow/build/libcst_transformers.py` - LibCST transformer classes
- `tests/test_libcst_transformers.py` - Comprehensive test suite

## Future Enhancements

- [ ] Update Field() metadata inside Annotated types (currently skipped)
- [ ] Parallel processing of multiple files
- [ ] Incremental updates (only process changed files)
- [ ] Custom prompt templates
- [ ] Fine-tuned models for workflow-specific generation
- [ ] Integration with workflow testing to validate generated content

## Contributing

Contributions are welcome! Please see the main FastWorkflow contributing guidelines.

## License

This component is part of FastWorkflow and is licensed under the Apache-2.0 License.
