import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import ast
import json
import shutil

from fastworkflow.build.genai_postprocessor import GenAIPostProcessor
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo
import libcst as cst


class TestGenAIPostProcessorUpdated(unittest.TestCase):
    """Test the updated GenAI postprocessor with LibCST-based implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal workflow structure
        self.workflow_path = os.path.join(self.temp_dir, "test_workflow")
        os.makedirs(os.path.join(self.workflow_path, "_commands"), exist_ok=True)
        os.makedirs(os.path.join(self.workflow_path, "___command_info"), exist_ok=True)
        
        # Create a test command file
        self.command_file = os.path.join(self.workflow_path, "_commands", "test_command.py")
        with open(self.command_file, 'w') as f:
            f.write("""
import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Signature:
    class Input(BaseModel):
        param1: str
        param2: int
    
    class Output(BaseModel):
        result: bool
    
    plain_utterances = []
    
    template_utterances = []
    
    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [command_name.split('/')[-1].lower().replace('_', ' ')]

class ResponseGenerator:
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        return Signature.Output(result=True)
    
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=output.model_dump_json())
            ]
        )
""")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_class_info(self):
        """Create a test ClassInfo object."""
        class_info = ClassInfo("TestClass", "test_module.py")
        method_info = MethodInfo(
            name="test_method",
            parameters=[
                {"name": "param1", "annotation": "str"},
                {"name": "param2", "annotation": "int"}
            ],
            docstring="Test method docstring",
            return_annotation="bool"
        )
        class_info.methods.append(method_info)
        return class_info
    
    @patch('fastworkflow.get_env_var')
    def test_initialization(self, mock_get_env):
        """Test that the processor initializes correctly."""
        mock_get_env.return_value = 'test_value'
        
        with patch.object(GenAIPostProcessor, '_initialize_dspy'):
            processor = GenAIPostProcessor()
            
            # Check that modules are initialized
            self.assertIsNotNone(processor.field_generator)
            self.assertIsNotNone(processor.utterance_generator)
            self.assertIsNotNone(processor.docstring_generator)
            self.assertIsNotNone(processor.workflow_generator)
            
            # Check that stats are initialized
            self.assertIn('files_processed', processor.stats)
            self.assertIn('files_updated', processor.stats)
    
    @patch('fastworkflow.get_env_var')
    def test_extract_current_state(self, mock_get_env):
        """Test extraction of current state from a command file."""
        mock_get_env.return_value = 'test_value'
        
        with patch.object(GenAIPostProcessor, '_initialize_dspy'):
            processor = GenAIPostProcessor()
            
            # Parse the test command file
            with open(self.command_file, 'r') as f:
                content = f.read()
            module = cst.parse_module(content)
            
            # Extract current state
            state = processor._extract_current_state(module)
            
            # Verify extracted state
            self.assertIn('input_fields', state)
            self.assertIn('output_fields', state)
            self.assertIn('plain_utterances', state)
            self.assertEqual(len(state['input_fields']), 2)
            self.assertEqual(len(state['output_fields']), 1)
            
            # Check field details
            input_fields = state['input_fields']
            self.assertEqual(input_fields[0]['name'], 'param1')
            self.assertEqual(input_fields[0]['type'], 'str')
            self.assertEqual(input_fields[1]['name'], 'param2')
            self.assertEqual(input_fields[1]['type'], 'int')
    
    @patch('fastworkflow.get_env_var')
    def test_generate_enhanced_content(self, mock_get_env):
        """Test generation of enhanced content for a command file."""
        mock_get_env.return_value = 'test_value'
        
        with patch.object(GenAIPostProcessor, '_initialize_dspy'):
            processor = GenAIPostProcessor()
            
            # Mock the DSPy modules
            processor.docstring_generator.generate_signature_docstring = MagicMock(
                return_value="Enhanced docstring"
            )
            processor.field_generator = MagicMock(
                return_value=MagicMock(
                    description="Enhanced description",
                    examples=["example1", "example2"],
                    pattern=""
                )
            )
            processor.utterance_generator = MagicMock(
                return_value=["new utterance 1", "new utterance 2"]
            )
            
            # Create current state
            current_state = {
                'has_signature_docstring': False,
                'input_fields': [
                    {'name': 'param1', 'type': 'str', 'has_description': False, 'has_examples': False},
                    {'name': 'param2', 'type': 'int', 'has_description': False, 'has_examples': False}
                ],
                'output_fields': [
                    {'name': 'result', 'type': 'bool', 'has_description': False, 'has_examples': False}
                ],
                'plain_utterances': []
            }
            
            # Generate enhanced content
            enhanced_data = processor._generate_enhanced_content(
                current_state, 
                "test_command", 
                "TestContext", 
                MagicMock(docstring="Test method docstring")
            )
            
            # Verify enhanced data
            self.assertIn('signature_docstring', enhanced_data)
            self.assertIn('field_metadata', enhanced_data)
            self.assertIn('new_utterances', enhanced_data)
            
            # Check field metadata
            field_metadata = enhanced_data['field_metadata']
            self.assertIn('Input.param1', field_metadata)
            self.assertIn('Input.param2', field_metadata)
            self.assertIn('Output.result', field_metadata)
    
    @patch('fastworkflow.get_env_var')
    def test_process_command_file_targeted(self, mock_get_env):
        """Test processing a command file with targeted updates."""
        mock_get_env.return_value = 'test_value'
        
        with patch.object(GenAIPostProcessor, '_initialize_dspy'), \
             patch.object(GenAIPostProcessor, '_extract_current_state') as mock_extract, \
             patch.object(GenAIPostProcessor, '_generate_enhanced_content') as mock_generate:
            
            # Setup mocks
            mock_extract.return_value = {
                'has_signature_docstring': False,
                'input_fields': [
                    {'name': 'param1', 'type': 'str', 'has_description': False, 'has_examples': False}
                ],
                'output_fields': [
                    {'name': 'result', 'type': 'bool', 'has_description': False, 'has_examples': False}
                ],
                'plain_utterances': []
            }
            
            mock_generate.return_value = {
                'signature_docstring': 'Enhanced docstring',
                'field_metadata': {
                    'Input.param1': {'description': 'Enhanced description', 'examples': ['example1']}
                },
                'new_utterances': ['new utterance 1']
            }
            
            # Create processor
            processor = GenAIPostProcessor()
            
            # Process command file
            result = processor._process_command_file_targeted(
                self.command_file, 
                "test_command", 
                "TestContext", 
                MagicMock(docstring="Test method docstring")
            )
            
            # Verify result
            self.assertTrue(result)
            self.assertEqual(processor.stats['files_processed'], 1)
    
    @patch('fastworkflow.get_env_var')
    def test_process_workflow(self, mock_get_env):
        """Test processing an entire workflow."""
        mock_get_env.return_value = 'test_value'
        
        # Create context directory
        context_dir = os.path.join(self.workflow_path, "_commands", "TestContext")
        os.makedirs(context_dir, exist_ok=True)
        
        # Create context command file
        context_command_file = os.path.join(context_dir, "test_context_command.py")
        shutil.copy(self.command_file, context_command_file)
        
        # Create context handler file
        context_handler_file = os.path.join(context_dir, "_TestContext.py")
        with open(context_handler_file, 'w') as f:
            f.write("# Context handler file\n")
        
        # Create classes dict
        classes = {
            "TestContext": self._create_test_class_info()
        }
        
        with patch.object(GenAIPostProcessor, '_initialize_dspy'), \
             patch.object(GenAIPostProcessor, '_process_command_file_targeted') as mock_process, \
             patch.object(GenAIPostProcessor, '_generate_context_handler_docstring') as mock_context_doc, \
             patch.object(GenAIPostProcessor, '_generate_workflow_description') as mock_workflow_doc:
            
            # Setup mocks
            mock_process.return_value = True
            
            # Create processor
            processor = GenAIPostProcessor()
            
            # Process workflow
            result = processor.process_workflow(self.workflow_path, classes)
            
            # Verify result
            self.assertTrue(result)
            self.assertTrue(mock_process.called)
            self.assertTrue(mock_context_doc.called)
            self.assertTrue(mock_workflow_doc.called)
    
    @patch('fastworkflow.get_env_var')
    def test_generate_context_handler_docstring(self, mock_get_env):
        """Test generating docstring for context handler file."""
        mock_get_env.return_value = 'test_value'
        
        # Create context directory and handler file
        context_dir = os.path.join(self.workflow_path, "_commands", "TestContext")
        os.makedirs(context_dir, exist_ok=True)
        
        context_handler_file = os.path.join(context_dir, "_TestContext.py")
        with open(context_handler_file, 'w') as f:
            f.write("""
class TestContextHandler:
    \"\"\"Original docstring\"\"\"
    def __init__(self):
        pass
""")
        
        with patch.object(GenAIPostProcessor, '_initialize_dspy'):
            processor = GenAIPostProcessor()
            
            # Mock docstring generator
            processor.docstring_generator.generate_context_docstring = MagicMock(
                return_value="Enhanced context docstring"
            )
            
            # Generate context handler docstring
            processor._generate_context_handler_docstring(
                context_dir, 
                "TestContext", 
                [{"name": "test_command", "docstring": "Test command docstring"}]
            )
            
            # Verify file was updated
            with open(context_handler_file, 'r') as f:
                content = f.read()
                self.assertIn("Enhanced context docstring", content)
    
    @patch('fastworkflow.get_env_var')
    def test_generate_workflow_description(self, mock_get_env):
        """Test generating workflow description file."""
        mock_get_env.return_value = 'test_value'
        
        with patch.object(GenAIPostProcessor, '_initialize_dspy'):
            processor = GenAIPostProcessor()
            
            # Mock workflow generator
            processor.workflow_generator = MagicMock(
                return_value="Enhanced workflow description"
            )
            
            # Generate workflow description
            processor._generate_workflow_description(
                self.workflow_path,
                {"TestContext": {"context_name": "TestContext", "commands": [], "docstring": ""}},
                []
            )
            
            # Verify file was created
            desc_file = os.path.join(self.workflow_path, "workflow_description.txt")
            self.assertTrue(os.path.exists(desc_file))
            
            # Verify content
            with open(desc_file, 'r') as f:
                content = f.read()
                self.assertEqual(content, "Enhanced workflow description")


if __name__ == '__main__':
    unittest.main()
