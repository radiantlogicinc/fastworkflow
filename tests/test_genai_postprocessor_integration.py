"""Integration tests for the GenAI post-processor component.

These tests verify the complete post-processing workflow including:
- Command file enhancement
- Context handler docstring generation
- Workflow description generation
"""

import os
import tempfile
import shutil
import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import ast
import dspy

from fastworkflow.build.genai_postprocessor import GenAIPostProcessor
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, FunctionInfo


class TestGenAIPostProcessorIntegration(unittest.TestCase):
    """Integration tests for the GenAI post-processor."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.workflow_path = os.path.join(self.temp_dir, "workflow")
        self.commands_dir = os.path.join(self.workflow_path, "_commands")
        self.context_dir = os.path.join(self.commands_dir, "TestClass")
        os.makedirs(self.context_dir)
        
        # Create sample command file
        self.command_file = os.path.join(self.context_dir, "test_method.py")
        self._create_sample_command_file(self.command_file)
        
        # Create sample context handler file
        self.context_handler = os.path.join(self.context_dir, "_TestClass.py")
        self._create_sample_context_handler(self.context_handler)
        
        # Create sample global command
        self.global_command = os.path.join(self.commands_dir, "global_function.py")
        self._create_sample_command_file(self.global_command)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_command_file(self, file_path):
        """Create a sample command file for testing."""
        content = '''
from pydantic import BaseModel, Field
import fastworkflow
from fastworkflow import Workflow

class Signature:
    class Input(BaseModel):
        param1: str
        param2: int
    
    class Output(BaseModel):
        result: bool
        message: str
    
    plain_utterances = ["test utterance"]
    template_utterances = []
    
    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return Signature.plain_utterances
    
    def validate_extracted_parameters(self, workflow: Workflow, command: str, cmd_parameters) -> None:
        pass

class ResponseGenerator:
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        """Process the test command."""
        return Signature.Output(result=True, message="Success")
'''
        with open(file_path, 'w') as f:
            f.write(content)
    
    def _create_sample_context_handler(self, file_path):
        """Create a sample context handler file for testing."""
        content = '''
# Context handler for TestClass
'''
        with open(file_path, 'w') as f:
            f.write(content)
    
    @patch('fastworkflow.get_env_var')
    def test_full_workflow_processing(self, mock_get_env):
        """Test the complete workflow processing."""
        # Setup mock environment variables
        mock_get_env.return_value = 'test_value'
        
        # Create test data
        classes = {
            "TestClass": self._create_test_class_info()
        }
        functions = {
            "global_function": self._create_test_function_info()
        }
        
        # Mock DSPy modules to return predictable results
        with patch.object(GenAIPostProcessor, '_initialize_dspy'), \
             patch('fastworkflow.build.genai_postprocessor.FieldMetadataGenerator') as mock_field_gen, \
             patch('fastworkflow.build.genai_postprocessor.UtteranceGenerator') as mock_utterance_gen, \
             patch('fastworkflow.build.genai_postprocessor.DocstringGenerator') as mock_docstring_gen, \
             patch('fastworkflow.build.genai_postprocessor.WorkflowDescriptionGenerator') as mock_workflow_gen:
            
            # Setup mocks
            self._setup_generator_mocks(
                mock_field_gen, mock_utterance_gen, 
                mock_docstring_gen, mock_workflow_gen
            )
            
            # Create processor
            processor = GenAIPostProcessor()
            
            # Process workflow
            result = processor.process_workflow(self.workflow_path, classes, functions)
            
            # Verify success
            self.assertTrue(result)
            
            # Verify command file was updated
            self._verify_command_file_updated(self.command_file)
            
            # Verify context handler was updated
            self._verify_context_handler_updated(self.context_handler)
            
            # Verify workflow description was created
            self._verify_workflow_description_created()
    
    def _create_test_class_info(self):
        """Create a test ClassInfo object."""
        class_info = ClassInfo(
            name="TestClass",
            module_path="test_module.py",
            docstring="Test class for testing"
        )
        
        method_info = MethodInfo(
            name="test_method",
            parameters=[
                {"name": "self"},
                {"name": "param1", "annotation": "str"},
                {"name": "param2", "annotation": "int"}
            ],
            docstring="Test method for processing",
            return_annotation="bool"
        )
        class_info.methods.append(method_info)
        
        return class_info
    
    def _create_test_function_info(self):
        """Create a test FunctionInfo object."""
        return FunctionInfo(
            name="global_function",
            module_path="test_module.py",
            parameters=[
                {"name": "param1", "annotation": "str"},
                {"name": "param2", "annotation": "int"}
            ],
            docstring="Global function for testing",
            return_annotation="bool"
        )
    
    def _setup_generator_mocks(self, mock_field_gen, mock_utterance_gen, 
                              mock_docstring_gen, mock_workflow_gen):
        """Setup mock generators to return predictable results."""
        # Mock field generator
        field_gen_instance = MagicMock()
        field_gen_instance.return_value = MagicMock(
            description="Enhanced field description",
            examples=["example1", "example2"],
            pattern="^[A-Za-z0-9]+$"
        )
        mock_field_gen.return_value = field_gen_instance
        
        # Mock utterance generator
        utterance_gen_instance = MagicMock()
        utterance_gen_instance.return_value = [
            "enhanced utterance 1",
            "enhanced utterance 2"
        ]
        mock_utterance_gen.return_value = utterance_gen_instance
        
        # Mock docstring generator
        docstring_gen_instance = MagicMock()
        docstring_gen_instance.generate_signature_docstring.return_value = "Enhanced signature docstring"
        docstring_gen_instance.generate_context_docstring.return_value = "Enhanced context docstring"
        mock_docstring_gen.return_value = docstring_gen_instance
        
        # Mock workflow generator
        workflow_gen_instance = MagicMock()
        workflow_gen_instance.return_value = "Enhanced workflow description"
        mock_workflow_gen.return_value = workflow_gen_instance
    
    def _verify_command_file_updated(self, file_path):
        """Verify that a command file was updated with enhanced content."""
        with open(file_path, 'r') as f:
            content = f.read()

        # Parse the updated file
        tree = ast.parse(content)

        sig_class = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Signature"
            ),
            None,
        )
        self.assertIsNotNone(sig_class, "Signature class should exist")

        # Check for docstring (should be first element if added)
        if sig_class.body and isinstance(sig_class.body[0], ast.Expr) and isinstance(sig_class.body[0].value, ast.Constant):
            self.assertIsInstance(sig_class.body[0].value.value, str)
    
    def _verify_context_handler_updated(self, file_path):
        """Verify that a context handler file was updated with docstring."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the updated file
        tree = ast.parse(content)
        
        # Check for module docstring
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            self.assertIsInstance(tree.body[0].value.value, str)
    
    def _verify_workflow_description_created(self):
        """Verify that workflow_description.txt was created."""
        desc_file = os.path.join(self.workflow_path, "workflow_description.txt")
        self.assertTrue(os.path.exists(desc_file), "workflow_description.txt should exist")
        
        with open(desc_file, 'r') as f:
            content = f.read()
        
        self.assertIn("workflow", content.lower(), "Should contain workflow description")
    
    @patch('fastworkflow.get_env_var')
    def test_error_recovery(self, mock_get_env):
        """Test that post-processor handles errors gracefully."""
        # Setup mock environment variables
        mock_get_env.return_value = 'test_value'
        
        # Create invalid command file
        invalid_file = os.path.join(self.context_dir, "invalid.py")
        with open(invalid_file, 'w') as f:
            f.write("invalid python syntax {")
        
        classes = {"TestClass": self._create_test_class_info()}
        
        # Create processor with mocked DSPy
        with patch.object(GenAIPostProcessor, '_initialize_dspy'):
            processor = GenAIPostProcessor()
            
            # Process should not crash on invalid files
            with patch('fastworkflow.build.genai_postprocessor.logger') as mock_logger:
                result = processor.process_workflow(self.workflow_path, classes, {})
                
                # Should still return True (doesn't fail the build)
                self.assertTrue(result)
                
                # Should log errors
                self.assertTrue(mock_logger.error.called)
    
    @patch('fastworkflow.get_env_var')
    def test_caching_behavior(self, mock_get_env):
        """Test that the processor uses caching to avoid redundant API calls."""
        # Setup mock environment variables
        mock_get_env.return_value = 'test_value'
        
        classes = {"TestClass": self._create_test_class_info()}
        
        with patch.object(GenAIPostProcessor, '_initialize_dspy'):
            processor = GenAIPostProcessor()
            
            # Test that cache is initialized
            self.assertIsInstance(processor.cache, dict)
            
            # Process workflow
            with patch.object(processor.field_generator, 'forward') as mock_field_forward:
                mock_field_forward.return_value = MagicMock(
                    description="Cached description",
                    examples=[],
                    constraints=""
                )
                
                # Process same field twice (simulated)
                processor._process_command_file(
                    self.command_file, "test_method", "TestClass", 
                    classes["TestClass"].methods[0]
                )
                
                # Verify field generator was called
                self.assertTrue(mock_field_forward.called)
    
    def test_command_file_ast_preservation(self):
        """Test that AST manipulation preserves the command file structure."""
        # Read original file
        with open(self.command_file, 'r') as f:
            original_content = f.read()
        
        original_tree = ast.parse(original_content)
        
        # Create processor with mocked env vars
        with patch('fastworkflow.get_env_var') as mock_env:
            mock_env.return_value = 'test_value'
            processor = GenAIPostProcessor()
        
        # Update the AST
        enhanced_input = [{'name': 'param1', 'type': 'str', 'description': 'Test', 'examples': [], 'pattern': ''}]
        enhanced_output = [{'name': 'result', 'type': 'bool', 'description': 'Test', 'examples': [], 'pattern': ''}]
        utterances = ["new utterance"]
        docstring = "Test docstring"
        
        updated_tree = processor._update_command_file_ast(
            original_tree, enhanced_input, enhanced_output, utterances, docstring
        )
        
        # Verify essential classes are preserved
        class_names = [node.name for node in ast.walk(updated_tree) if isinstance(node, ast.ClassDef)]
        self.assertIn("Signature", class_names)
        self.assertIn("ResponseGenerator", class_names)
        
        # Verify methods are preserved
        for node in ast.walk(updated_tree):
            if isinstance(node, ast.ClassDef) and node.name == "ResponseGenerator":
                method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                self.assertIn("_process_command", method_names)


class TestPostProcessorCLIIntegration(unittest.TestCase):
    """Test CLI integration for the post-processor."""
    
    def test_cli_arguments_parsing(self):
        """Test that CLI arguments are properly parsed."""
        from fastworkflow.build.__main__ import parse_args
        
        with patch('sys.argv', ['prog', '--app-dir', '/app', '--workflow-folderpath', '/workflow']):
            args = parse_args()
            self.assertEqual(args.app_dir, '/app')
            self.assertEqual(args.workflow_folderpath, '/workflow')
            # No more genai-related arguments
            self.assertFalse(hasattr(args, 'no_genai'))
            self.assertFalse(hasattr(args, 'genai_model'))
            self.assertFalse(hasattr(args, 'genai_api_key'))
    
    @patch('fastworkflow.build.__main__.run_genai_postprocessor')
    def test_build_flow_integration(self, mock_run_postprocessor):
        """Test that post-processor is called in the build flow."""
        from fastworkflow.build.__main__ import run_command_generation
        
        # Create mock args
        args = MagicMock()
        args.app_dir = "/test/app"
        args.workflow_folderpath = "/test/workflow"
        args.overwrite = False
        args.stub_commands = None
        args.no_startup = False
        
        # Mock dependencies
        with patch('fastworkflow.build.__main__.ast_class_extractor.analyze_python_file') as mock_analyze, \
             patch('fastworkflow.build.__main__.real_generate_context_model') as mock_gen_model, \
             patch('fastworkflow.build.__main__.real_generate_command_files') as mock_gen_files, \
             patch('fastworkflow.build.__main__.generate_startup_command') as mock_gen_startup, \
             patch('fastworkflow.build.__main__.ContextFolderGenerator') as mock_folder_gen, \
             patch('fastworkflow.build.__main__.NavigatorStubGenerator') as mock_nav_gen, \
             patch('glob.glob') as mock_glob, \
             patch('os.path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open:
            
            # Setup mocks
            mock_glob.return_value = []
            mock_analyze.return_value = ({}, {})
            mock_gen_model.return_value = {}
            mock_exists.return_value = False
            mock_folder_gen.return_value.generate_folders.return_value = {}
            mock_nav_gen.return_value.generate_navigator_stubs.return_value = {}
            
            # Run command generation
            all_classes, context_model = run_command_generation(args)
            
            # Verify post-processor was called
            mock_run_postprocessor.assert_called_once_with(args, {}, {})
    



if __name__ == '__main__':
    unittest.main()
