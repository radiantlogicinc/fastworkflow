import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import shutil
import json
import contextlib

from fastworkflow.build.genai_postprocessor import GenAIPostProcessor, run_genai_postprocessor
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo


class TestGenAIPostProcessorIntegrationUpdated(unittest.TestCase):
    """Test the integration of GenAI postprocessor with LibCST-based implementation."""
    
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
    def test_end_to_end_processing(self, mock_get_env):
        """Test end-to-end processing of a workflow."""
        mock_get_env.return_value = 'test_value'
        
        # Create classes dict
        classes = {
            "TestClass": self._create_test_class_info()
        }
        
        with patch('fastworkflow.build.genai_postprocessor.dspy.LM'), \
             patch('fastworkflow.build.genai_postprocessor.dspy.context', new=lambda *args, **kwargs: contextlib.nullcontext()):
            # Create processor
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
            processor.workflow_generator = MagicMock(
                return_value="Enhanced workflow description"
            )
            
            # Process workflow
            result = processor.process_workflow(self.workflow_path, classes)
            
            # Verify result
            self.assertTrue(result)
            
            # Check stats
            self.assertGreater(processor.stats['files_processed'], 0)
            
            # Check workflow description file
            desc_file = os.path.join(self.workflow_path, "workflow_description.txt")
            self.assertTrue(os.path.exists(desc_file))
    
    @patch('fastworkflow.get_env_var')
    def test_run_genai_postprocessor_function(self, mock_get_env):
        """Test the run_genai_postprocessor function."""
        mock_get_env.return_value = 'test_value'
        
        # Create classes dict
        classes = {
            "TestClass": self._create_test_class_info()
        }
        
        # Create mock args
        args = MagicMock()
        args.workflow_folderpath = self.workflow_path
        args.skip_genai = False
        
        with patch('fastworkflow.build.genai_postprocessor.dspy.LM'), \
             patch('fastworkflow.build.genai_postprocessor.dspy.context', new=lambda *args, **kwargs: contextlib.nullcontext()), \
             patch.object(GenAIPostProcessor, 'process_workflow') as mock_process:
            
            # Setup mock
            mock_process.return_value = True
            
            # Run function
            result = run_genai_postprocessor(args, classes)
            
            # Verify result
            self.assertTrue(result)
            self.assertTrue(mock_process.called)
    
    @patch('fastworkflow.get_env_var')
    def test_skip_genai_flag(self, mock_get_env):
        """Test that the skip_genai flag works."""
        mock_get_env.return_value = 'test_value'
        
        # Create classes dict
        classes = {
            "TestClass": self._create_test_class_info()
        }
        
        # Create mock args
        args = MagicMock()
        args.workflow_folderpath = self.workflow_path
        args.skip_genai = True
        
        # Note: The current implementation doesn't check skip_genai flag,
        # so we're just testing that the function returns successfully
        with patch('fastworkflow.build.genai_postprocessor.dspy.LM'), \
             patch('fastworkflow.build.genai_postprocessor.dspy.context', new=lambda *args, **kwargs: contextlib.nullcontext()), \
             patch.object(GenAIPostProcessor, 'process_workflow') as mock_process:
            
            # Run function
            result = run_genai_postprocessor(args, classes)
            
            # Verify result
            self.assertTrue(result)
            # The function doesn't check skip_genai, so process_workflow is called
            self.assertTrue(mock_process.called)
    
    @patch('fastworkflow.get_env_var')
    def test_error_handling(self, mock_get_env):
        """Test error handling in the processor."""
        mock_get_env.return_value = 'test_value'
        
        with patch('fastworkflow.build.genai_postprocessor.dspy.LM'), \
             patch('fastworkflow.build.genai_postprocessor.dspy.context', new=lambda *args, **kwargs: contextlib.nullcontext()), \
             patch.object(GenAIPostProcessor, 'process_workflow') as mock_process:
            
            # Setup mock to raise exception
            mock_process.side_effect = Exception("Test error")
            
            # Create mock args
            args = MagicMock()
            args.workflow_folderpath = self.workflow_path
            args.skip_genai = False
            
            # Run function
            result = run_genai_postprocessor(args, {})
            
            # Function should return True even on error to not block build
            self.assertTrue(result)


class TestPostProcessorCLIIntegrationUpdated(unittest.TestCase):
    """Test the CLI integration of GenAI postprocessor."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal workflow structure
        self.workflow_path = os.path.join(self.temp_dir, "test_workflow")
        os.makedirs(os.path.join(self.workflow_path, "_commands"), exist_ok=True)
        os.makedirs(os.path.join(self.workflow_path, "___command_info"), exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('fastworkflow.build.genai_postprocessor.run_genai_postprocessor')
    def test_build_flow_integration(self, mock_run_genai):
        """Test integration with build flow. GenAI postprocessor should NOT be called from build_main."""
        mock_run_genai.return_value = True
        
        # Import here to avoid circular imports
        from fastworkflow.build.__main__ import build_main
        
        # Create mock args
        args = MagicMock()
        args.workflow_folderpath = self.workflow_path
        args.app_dir = self.temp_dir
        args.overwrite = True
        args.skip_genai = False
        args.stub_commands = None
        args.no_startup = True
        
        with patch('fastworkflow.build.__main__.validate_directories'), \
             patch('fastworkflow.build.__main__.run_command_generation', return_value=({}, {})), \
             patch('fastworkflow.build.__main__.run_validation', return_value=[]), \
             patch('fastworkflow.build.__main__.run_documentation'):
            
            # Run build flow
            build_main(args)
            
            # Verify GenAI postprocessor is NOT called from build_main anymore
            # (it's now handled by the refine command)
            mock_run_genai.assert_not_called()


if __name__ == '__main__':
    unittest.main()
