"""Unit tests for the GenAI post-processor component."""

import os
import ast
import json
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
from typing import Dict, List, Any

import dspy

from fastworkflow.build.genai_postprocessor import (
    FieldMetadataSignature,
    UtteranceGeneratorSignature,
    SignatureDocstringSignature,
    ContextDocstringSignature,
    WorkflowDescriptionSignature,
    FieldMetadataGenerator,
    UtteranceGenerator,
    DocstringGenerator,
    WorkflowDescriptionGenerator,
    GenAIPostProcessor,
    run_genai_postprocessor
)
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo, FunctionInfo


class TestDSPySignatures(unittest.TestCase):
    """Test DSPy signature definitions."""
    
    def test_field_metadata_signature(self):
        # sourcery skip: class-extract-method
        """Test FieldMetadataSignature has correct fields."""
        sig = FieldMetadataSignature
        
        # Check input fields
        self.assertIn('field_name', sig.__annotations__)
        self.assertIn('field_type', sig.__annotations__)
        self.assertIn('method_docstring', sig.__annotations__)
        self.assertIn('method_name', sig.__annotations__)
        self.assertIn('context_name', sig.__annotations__)
        self.assertIn('is_input', sig.__annotations__)
        
        # Check output fields
        self.assertIn('description', sig.__annotations__)
        self.assertIn('examples', sig.__annotations__)
        self.assertIn('pattern', sig.__annotations__)
    
    def test_utterance_generator_signature(self):
        """Test UtteranceGeneratorSignature has correct fields."""
        sig = UtteranceGeneratorSignature
        
        # Check input fields
        self.assertIn('command_name', sig.__annotations__)
        self.assertIn('command_docstring', sig.__annotations__)
        self.assertIn('command_input_fields', sig.__annotations__)
        
        # Check output fields
        self.assertIn('utterances', sig.__annotations__)
    
    def test_signature_docstring_signature(self):
        """Test SignatureDocstringSignature has correct fields."""
        sig = SignatureDocstringSignature
        
        # Check input fields
        self.assertIn('command_name', sig.__annotations__)
        self.assertIn('input_fields_json', sig.__annotations__)
        self.assertIn('output_fields_json', sig.__annotations__)
        self.assertIn('context_name', sig.__annotations__)
        self.assertIn('original_docstring', sig.__annotations__)
        
        # Check output fields
        self.assertIn('docstring', sig.__annotations__)
    
    def test_context_docstring_signature(self):
        """Test ContextDocstringSignature has correct fields."""
        sig = ContextDocstringSignature
        
        # Check input fields
        self.assertIn('context_name', sig.__annotations__)
        self.assertIn('commands_json', sig.__annotations__)
        
        # Check output fields
        self.assertIn('docstring', sig.__annotations__)
    
    def test_workflow_description_signature(self):
        """Test WorkflowDescriptionSignature has correct fields."""
        sig = WorkflowDescriptionSignature
        
        # Check input fields
        self.assertIn('contexts_json', sig.__annotations__)
        self.assertIn('global_commands_json', sig.__annotations__)
        
        # Check output fields
        self.assertIn('description', sig.__annotations__)


class TestDSPyModules(unittest.TestCase):
    """Test DSPy module implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock DSPy settings
        self.mock_lm = MagicMock()
        with patch('dspy.settings') as mock_settings:
            mock_settings.configure = MagicMock()
    
    @patch('dspy.ChainOfThought')
    def test_field_metadata_generator(self, mock_cot):
        """Test FieldMetadataGenerator module."""
        # Setup mock
        mock_generate = MagicMock()
        mock_result = MagicMock()
        mock_result.description = "Test field description"
        mock_result.examples = ["example1", "example2"]
        mock_result.pattern = "^\\d+$"
        mock_generate.return_value = mock_result
        mock_cot.return_value = mock_generate
        
        # Create generator
        generator = FieldMetadataGenerator()
        
        # Test forward method
        result = generator.forward(
            field_name="test_field",
            field_type="int",
            method_docstring="Test method",
            method_name="test_method",
            context_name="TestContext",
            is_input=True
        )
        
        # Verify results
        self.assertEqual(result.description, "Test field description")
        self.assertEqual(result.examples, ["example1", "example2"])
        self.assertEqual(result.pattern, "^\\d+$")
        
        # Verify call
        mock_generate.assert_called_once()
    
    @patch('dspy.ChainOfThought')
    def test_field_metadata_generator_error_handling(self, mock_cot):
        """Test FieldMetadataGenerator error handling."""
        # Setup mock to raise exception
        mock_generate = MagicMock(side_effect=Exception("API error"))
        mock_cot.return_value = mock_generate
        
        # Create generator
        generator = FieldMetadataGenerator()
        
        # Test forward method with error
        result = generator.forward(
            field_name="test_field",
            field_type="int",
            method_docstring="Test method",
            method_name="test_method",
            context_name="TestContext",
            is_input=True
        )
        
        # Should return default values
        self.assertEqual(result.description, "The test_field parameter")
        self.assertEqual(result.examples, [])
        self.assertEqual(result.pattern, "")
    
    @patch('dspy.ChainOfThought')
    def test_utterance_generator(self, mock_cot):
        """Test UtteranceGenerator module."""
        # Setup mock
        mock_generate = MagicMock()
        mock_result = MagicMock()
        mock_result.utterances = ["utterance 1", "utterance 2", "utterance 3"]
        mock_generate.return_value = mock_result
        mock_cot.return_value = mock_generate
        
        # Create generator
        generator = UtteranceGenerator()
        
        # Test forward method
        result = generator.forward(
            command_name="test_command",
            command_docstring="Test command description",
            input_fields=[
                {"name": "field1", "type": "str", "description": "Field 1"},
                {"name": "field2", "type": "int", "description": "Field 2"}
            ]
        )
        
        # Verify results
        self.assertEqual(result, ["utterance 1", "utterance 2", "utterance 3"])
    
    @patch('dspy.ChainOfThought')
    def test_utterance_generator_error_handling(self, mock_cot):
        """Test UtteranceGenerator error handling."""
        # Setup mock to raise exception
        mock_generate = MagicMock(side_effect=Exception("API error"))
        mock_cot.return_value = mock_generate
        
        # Create generator
        generator = UtteranceGenerator()
        
        # Test forward method with error
        result = generator.forward(
            command_name="test_command",
            command_docstring="Test command description",
            input_fields=[]
        )
        
        # Should return default utterance
        self.assertEqual(result, ["test command"])
    
    @patch('dspy.ChainOfThought')
    def test_docstring_generator_signature(self, mock_cot):
        """Test DocstringGenerator for signature docstrings."""
        # Setup mock
        mock_generate = MagicMock()
        mock_result = MagicMock()
        mock_result.docstring = "Generated signature docstring"
        mock_generate.return_value = mock_result
        mock_cot.return_value = mock_generate
        
        # Create generator
        generator = DocstringGenerator()
        
        # Test generate_signature_docstring method
        result = generator.generate_signature_docstring(
            command_name="test_command",
            input_fields=[{"name": "field1", "type": "str", "description": "Field 1"}],
            output_fields=[{"name": "result", "type": "bool", "description": "Result"}],
            context_name="TestContext",
            original_docstring="Original docstring"
        )
        
        # Verify results
        self.assertEqual(result, "Generated signature docstring")
    
    @patch('dspy.ChainOfThought')
    def test_docstring_generator_context(self, mock_cot):
        """Test DocstringGenerator for context docstrings."""
        # Setup mock
        mock_generate = MagicMock()
        mock_result = MagicMock()
        mock_result.docstring = "Generated context docstring"
        mock_generate.return_value = mock_result
        mock_cot.return_value = mock_generate
        
        # Create generator
        generator = DocstringGenerator()
        
        # Test generate_context_docstring method
        result = generator.generate_context_docstring(
            context_name="TestContext",
            commands=[
                {"name": "command1", "docstring": "Command 1"},
                {"name": "command2", "docstring": "Command 2"}
            ]
        )
        
        # Verify results
        self.assertEqual(result, "Generated context docstring")
    
    @patch('dspy.ChainOfThought')
    def test_workflow_description_generator(self, mock_cot):
        """Test WorkflowDescriptionGenerator module."""
        # Setup mock
        mock_generate = MagicMock()
        mock_result = MagicMock()
        mock_result.description = "Generated workflow description"
        mock_generate.return_value = mock_result
        mock_cot.return_value = mock_generate
        
        # Create generator
        generator = WorkflowDescriptionGenerator()
        
        # Test forward method
        result = generator.forward(
            contexts=[
                {
                    "context_name": "Context1",
                    "docstring": "Context 1 doc",
                    "commands": [{"name": "cmd1", "docstring": "Cmd 1"}]
                }
            ],
            global_commands=[
                {"name": "global_cmd", "docstring": "Global command"}
            ]
        )
        
        # Verify results
        self.assertEqual(result, "Generated workflow description")


class TestGenAIPostProcessor(unittest.TestCase):
    """Test the main GenAIPostProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workflow_path = os.path.join(self.temp_dir, "workflow")
        self.commands_dir = os.path.join(self.workflow_path, "_commands")
        os.makedirs(self.commands_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('fastworkflow.get_env_var')
    def test_initialization(self, mock_get_env):
        """Test GenAIPostProcessor initialization."""
        # Setup mock environment variables
        mock_get_env.side_effect = lambda key, default=None: {
            'LLM_COMMAND_METADATA_GEN': 'mistral/mistral-small-latest',
            'LITELLM_API_KEY_COMMANDMETADATA_GEN': 'test_key'
        }.get(key, default)
        
        # Create processor
        processor = GenAIPostProcessor()
        
        # Verify initialization
        self.assertEqual(processor.model, 'mistral/mistral-small-latest')
        self.assertEqual(processor.api_key, 'test_key')
        self.assertIsNotNone(processor.field_generator)
        self.assertIsNotNone(processor.utterance_generator)
        self.assertIsNotNone(processor.docstring_generator)
        self.assertIsNotNone(processor.workflow_generator)
    
    def test_extract_fields_from_class(self):
        """Test field extraction from AST class."""
        # Create a sample AST
        code = """
class Signature:
    class Input(BaseModel):
        field1: str
        field2: int = 5
        field3: List[str] = Field(description="Test")
    
    class Output(BaseModel):
        result: bool
        message: str
"""
        tree = ast.parse(code)
        signature_class = tree.body[0]
        
        # Create processor with mocked env vars
        with patch('fastworkflow.get_env_var') as mock_env:
            mock_env.return_value = 'test_value'
            processor = GenAIPostProcessor()
        
        # Test input field extraction
        input_fields = processor._extract_fields_from_class(signature_class, "Input")
        self.assertEqual(len(input_fields), 3)
        self.assertEqual(input_fields[0]['name'], 'field1')
        self.assertEqual(input_fields[0]['type'], 'str')
        self.assertEqual(input_fields[1]['name'], 'field2')
        self.assertEqual(input_fields[1]['type'], 'int')
        
        # Test output field extraction
        output_fields = processor._extract_fields_from_class(signature_class, "Output")
        self.assertEqual(len(output_fields), 2)
        self.assertEqual(output_fields[0]['name'], 'result')
        self.assertEqual(output_fields[0]['type'], 'bool')
    
    def test_update_model_fields(self):
        """Test updating model fields with enhanced metadata."""
        # Create a sample AST
        code = """
class Input(BaseModel):
    field1: str
    field2: int
"""
        tree = ast.parse(code)
        class_node = tree.body[0]
        
        # Enhanced fields data
        enhanced_fields = [
            {
                'name': 'field1',
                'type': 'str',
                'description': 'First field',
                'examples': ['example1', 'example2'],
                'pattern': '^[A-Za-z]+$'
            },
            {
                'name': 'field2',
                'type': 'int',
                'description': 'Second field',
                'examples': [1, 2, 3],
                'pattern': ''  # No pattern for integer
            }
        ]
        
        # Create processor with mocked env vars
        with patch('fastworkflow.get_env_var') as mock_env:
            mock_env.return_value = 'test_value'
            processor = GenAIPostProcessor()
        
        # Update fields
        processor._update_model_fields(class_node, enhanced_fields)
        
        # Verify the AST was updated
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'field1':
                self.assertIsInstance(node.value, ast.Call)
                self.assertEqual(node.value.func.id, 'Field')
                # Check for description keyword
                desc_kwarg = next((k for k in node.value.keywords if k.arg == 'description'), None)
                self.assertIsNotNone(desc_kwarg)
                self.assertEqual(desc_kwarg.value.value, 'First field')
    
    def test_update_command_file_ast(self):
        """Test updating command file AST with enhanced content."""
        # Create a sample command file AST
        code = """
class Signature:
    class Input(BaseModel):
        field1: str
    
    class Output(BaseModel):
        result: bool
    
    plain_utterances = ["old utterance"]
    template_utterances = []
"""
        tree = ast.parse(code)
        
        # Enhanced data
        enhanced_input = [{
            'name': 'field1',
            'type': 'str',
            'description': 'Input field',
            'examples': ['ex1'],
            'pattern': ''
        }]
        enhanced_output = [{
            'name': 'result',
            'type': 'bool',
            'description': 'Output field',
            'examples': [True, False],
            'pattern': ''
        }]
        utterances = ["new utterance 1", "new utterance 2"]
        docstring = "Enhanced docstring"
        
        # Create processor with mocked env vars
        with patch('fastworkflow.get_env_var') as mock_env:
            mock_env.return_value = 'test_value'
            processor = GenAIPostProcessor()
        
        # Update AST
        updated_tree = processor._update_command_file_ast(
            tree, enhanced_input, enhanced_output, utterances, docstring
        )
        
        # Verify signature class has docstring
        sig_class = updated_tree.body[0]
        self.assertIsInstance(sig_class.body[0], ast.Expr)
        self.assertEqual(sig_class.body[0].value.value, "Enhanced docstring")
        
        # Verify utterances were updated
        for node in sig_class.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "plain_utterances":
                        self.assertIsInstance(node.value, ast.List)
                        self.assertEqual(len(node.value.elts), 2)
                        self.assertEqual(node.value.elts[0].value, "new utterance 1")
    
    @patch('fastworkflow.build.genai_postprocessor.logger')
    def test_process_command_file_no_signature_class(self, mock_logger):
        """Test processing command file without Signature class."""
        # Create a command file without Signature class
        file_path = os.path.join(self.commands_dir, "test_command.py")
        with open(file_path, 'w') as f:
            f.write("# Empty file\n")
        
        # Create processor with mocked env vars
        with patch('fastworkflow.get_env_var') as mock_env:
            mock_env.return_value = 'test_value'
            processor = GenAIPostProcessor()
        
        # Process file
        result = processor._process_command_file(file_path, "test_command", None, None)
        
        # Should return False and log warning
        self.assertFalse(result)
        mock_logger.warning.assert_called()
    
    def test_generate_workflow_description(self):
        """Test workflow description generation."""
        # Create processor with default env
        with patch('fastworkflow.get_env_var') as mock_env:
            mock_env.side_effect = lambda key, default=None: default
            processor = GenAIPostProcessor()
        
        # Mock the workflow generator
        processor.workflow_generator = MagicMock()
        processor.workflow_generator.return_value = "Test workflow description"
        
        # Test data
        contexts = {
            "Context1": {
                "context_name": "Context1",
                "commands": [{"name": "cmd1", "docstring": "Command 1"}],
                "docstring": "Context 1"
            }
        }
        global_commands = [{"name": "global_cmd", "docstring": "Global"}]
        
        # Generate description
        processor._generate_workflow_description(self.workflow_path, contexts, global_commands)
        
        # Verify file was created
        desc_file = os.path.join(self.workflow_path, "workflow_description.txt")
        self.assertTrue(os.path.exists(desc_file))
        
        # Verify content
        with open(desc_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, "Test workflow description")


class TestIntegrationFunction(unittest.TestCase):
    """Test the integration function for the build tool."""
    
    @patch('fastworkflow.build.genai_postprocessor.GenAIPostProcessor')
    def test_run_genai_postprocessor_success(self, mock_processor_class):
        """Test successful post-processing run."""
        # Create mock args
        args = MagicMock()
        args.workflow_folderpath = "/test/path"
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.process_workflow.return_value = True
        mock_processor_class.return_value = mock_processor
        
        # Test data
        classes = {"TestClass": MagicMock()}
        functions = {"test_func": MagicMock()}
        
        # Run post-processor
        result = run_genai_postprocessor(args, classes, functions)
        
        # Verify
        self.assertTrue(result)
        mock_processor_class.assert_called_once_with()
        mock_processor.process_workflow.assert_called_once_with("/test/path", classes, functions)
    
    @patch('fastworkflow.build.genai_postprocessor.GenAIPostProcessor')
    @patch('fastworkflow.build.genai_postprocessor.logger')
    def test_run_genai_postprocessor_exception(self, mock_logger, mock_processor_class):
        """Test exception handling in post-processor."""
        # Create mock args
        args = MagicMock()
        args.workflow_folderpath = "/test/path"
        
        # Mock processor to raise exception
        mock_processor_class.side_effect = Exception("Test error")
        
        # Run post-processor
        result = run_genai_postprocessor(args, {}, {})
        
        # Should return True (doesn't fail the build)
        self.assertTrue(result)
        mock_logger.error.assert_called()


if __name__ == '__main__':
    unittest.main()
