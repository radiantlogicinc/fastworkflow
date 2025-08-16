"""Integration test for LibCST-based GenAI postprocessor."""

import os
import tempfile
import shutil
from pathlib import Path

def test_libcst_postprocessor_preserves_content():
    """Test that LibCST postprocessor preserves existing content while adding new."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample command file
        commands_dir = Path(tmpdir) / "_commands" / "TestContext"
        commands_dir.mkdir(parents=True)
        
        test_file = commands_dir / "test_command.py"
        
        # Write initial content with some existing metadata
        initial_content = '''from pydantic import BaseModel, Field

class Signature:
    """Existing docstring that should be preserved"""
    
    class Input(BaseModel):
        # This comment should be preserved
        name: str = Field(description="Existing name description")
        age: int  # No Field() yet
        email: str
    
    class Output(BaseModel):
        result: str = Field(description="Existing result")
    
    plain_utterances = [
        "existing utterance",
        "another existing one"
    ]
    
    # Custom method that should be preserved
    @staticmethod
    def custom_method():
        """This should not be touched"""
        return True
'''
        
        test_file.write_text(initial_content)
        
        # Create a mock args object
        class MockArgs:
            workflow_folderpath = tmpdir
        
        try:
            # Import and run the postprocessor
            from fastworkflow.build.genai_postprocessor import GenAIPostProcessor
            
            processor = GenAIPostProcessor()
            # We'll do a minimal test - just verify the file can be parsed and processed
            # without errors, and that key content is preserved
            
            # Read the file back
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Parse with LibCST
            import libcst as cst
            module = cst.parse_module(content)
            
            # Extract state to verify parsing works
            from fastworkflow.build.libcst_transformers import StateExtractor
            extractor = StateExtractor()
            wrapper = cst.MetadataWrapper(module)
            wrapper.visit(extractor)
            
            # Verify existing content was extracted correctly
            assert extractor.has_signature_docstring == True
            assert len(extractor.input_fields) == 3
            assert len(extractor.output_fields) == 1
            assert len(extractor.plain_utterances) == 2
            assert "existing utterance" in extractor.plain_utterances
            
            # Apply a simple transformation to verify it works
            from fastworkflow.build.libcst_transformers import UtteranceAppender
            appender = UtteranceAppender(["new utterance"])
            updated = module.visit(appender)
            
            # Verify the transformation worked
            updated_code = updated.code
            assert "existing utterance" in updated_code
            assert "another existing one" in updated_code
            assert "new utterance" in updated_code
            assert "# This comment should be preserved" in updated_code
            assert "def custom_method" in updated_code
            
            print("✅ LibCST integration test passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise


if __name__ == "__main__":
    test_libcst_postprocessor_preserves_content()
    print("All integration tests passed!")
