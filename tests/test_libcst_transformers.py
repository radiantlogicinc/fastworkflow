"""Tests for LibCST transformers used in targeted command file updates."""

import unittest
import libcst as cst
from fastworkflow.build.libcst_transformers import (
    SignatureDocstringUpdater,
    FieldMetadataUpdater,
    UtteranceAppender,
    StateExtractor
)


class TestSignatureDocstringUpdater(unittest.TestCase):
    """Test the SignatureDocstringUpdater transformer."""
    
    def test_add_docstring_when_missing(self):
        """Test that docstring is added when missing."""
        code = '''class Signature:
    class Input(BaseModel):
        pass'''
        
        module = cst.parse_module(code)
        transformer = SignatureDocstringUpdater("New docstring for the command")
        updated = module.visit(transformer)
        
        self.assertTrue(transformer.signature_updated)
        self.assertIn('"""New docstring for the command"""', updated.code)
    
    def test_preserve_existing_docstring(self):
        """Test that existing docstring is preserved."""
        code = '''class Signature:
    """Existing docstring"""
    class Input(BaseModel):
        pass'''
        
        module = cst.parse_module(code)
        transformer = SignatureDocstringUpdater("New docstring")
        updated = module.visit(transformer)
        
        self.assertFalse(transformer.signature_updated)
        self.assertIn('"""Existing docstring"""', updated.code)
        self.assertNotIn('New docstring', updated.code)
    
    def test_replace_empty_docstring(self):
        """Test that empty docstring is replaced."""
        code = '''class Signature:
    """"""
    class Input(BaseModel):
        pass'''
        
        module = cst.parse_module(code)
        transformer = SignatureDocstringUpdater("New docstring")
        updated = module.visit(transformer)
        
        self.assertTrue(transformer.signature_updated)
        self.assertIn('"""New docstring"""', updated.code)


class TestFieldMetadataUpdater(unittest.TestCase):
    """Test the FieldMetadataUpdater transformer."""
    
    def test_add_field_call_when_missing(self):
        """Test adding Field() call to fields without it."""
        code = '''class Signature:
    class Input(BaseModel):
        name: str
        age: int'''
        
        metadata = {
            "Input.name": {
                "description": "Person's name",
                "examples": ["John", "Jane"]
            },
            "Input.age": {
                "description": "Age in years",
                "examples": [25, 30]
            }
        }
        
        module = cst.parse_module(code)
        transformer = FieldMetadataUpdater(metadata)
        updated = module.visit(transformer)
        
        # Check that Field() was added with correct descriptions (LibCST adds spaces around =)
        self.assertIn('description', updated.code)
        self.assertIn('"Person\'s name"', updated.code)
        self.assertIn('"Age in years"', updated.code)
        self.assertIn('Field', updated.code)
        self.assertEqual(len(transformer.fields_updated), 2)
    
    def test_preserve_existing_field_metadata(self):
        """Test that existing Field() metadata is preserved."""
        code = '''class Signature:
    class Input(BaseModel):
        name: str = Field(description="Existing description")
        age: int'''
        
        metadata = {
            "Input.name": {
                "description": "New description",
                "examples": ["John", "Jane"]
            },
            "Input.age": {
                "description": "Age in years"
            }
        }
        
        module = cst.parse_module(code)
        transformer = FieldMetadataUpdater(metadata)
        updated = module.visit(transformer)
        
        # Existing description should be preserved
        self.assertIn('description="Existing description"', updated.code)
        self.assertNotIn('New description', updated.code)
        
        # Examples should be added to name field
        self.assertIn('"John"', updated.code)
        self.assertIn('"Jane"', updated.code)
        
        # Age should get Field() call
        self.assertIn('Field', updated.code)
        self.assertIn('"Age in years"', updated.code)
    
    def test_merge_partial_metadata(self):
        """Test merging new metadata with existing partial metadata."""
        code = '''class Signature:
    class Input(BaseModel):
        email: str = Field(description="Email address")'''
        
        metadata = {
            "Input.email": {
                "examples": ["user@example.com", "admin@site.org"],
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            }
        }
        
        module = cst.parse_module(code)
        transformer = FieldMetadataUpdater(metadata)
        updated = module.visit(transformer)
        
        # Should preserve description and add examples and pattern
        self.assertIn('description="Email address"', updated.code)
        self.assertIn('examples', updated.code)  # LibCST may add spaces around =
        self.assertIn('pattern', updated.code)


class TestUtteranceAppender(unittest.TestCase):
    """Test the UtteranceAppender transformer."""
    
    def test_append_new_utterances(self):
        """Test appending new utterances to existing list."""
        code = '''class Signature:
    plain_utterances = [
        "create project",
        "new project"
    ]'''
        
        new_utterances = ["create todo list", "make a new list"]
        
        module = cst.parse_module(code)
        transformer = UtteranceAppender(new_utterances)
        updated = module.visit(transformer)
        
        self.assertTrue(transformer.utterances_updated)
        # Original utterances preserved
        self.assertIn('"create project"', updated.code)
        self.assertIn('"new project"', updated.code)
        # New utterances added
        self.assertIn('"create todo list"', updated.code)
        self.assertIn('"make a new list"', updated.code)
    
    def test_no_duplicate_utterances(self):
        """Test that duplicate utterances are not added."""
        code = '''class Signature:
    plain_utterances = [
        "create project",
        "new project"
    ]'''
        
        new_utterances = ["create project", "new task", "new project"]
        
        module = cst.parse_module(code)
        transformer = UtteranceAppender(new_utterances)
        updated = module.visit(transformer)
        
        # Only "new task" should be added
        self.assertTrue(transformer.utterances_updated)
        self.assertIn('"new task"', updated.code)
        
        # Count occurrences of "create project"
        self.assertEqual(updated.code.count('"create project"'), 1)
        self.assertEqual(updated.code.count('"new project"'), 1)
    
    def test_preserve_quote_style(self):
        """Test that quote style is preserved."""
        code = """class Signature:
    plain_utterances = [
        'single quotes',
        'another one'
    ]"""
        
        new_utterances = ["new utterance"]
        
        module = cst.parse_module(code)
        transformer = UtteranceAppender(new_utterances)
        updated = module.visit(transformer)
        
        # Should use single quotes to match existing
        self.assertIn("'new utterance'", updated.code)
        self.assertNotIn('"new utterance"', updated.code)
    
    def test_empty_utterances_list(self):
        """Test handling empty utterances list."""
        code = '''class Signature:
    plain_utterances = []'''
        
        new_utterances = ["first utterance", "second utterance"]
        
        module = cst.parse_module(code)
        transformer = UtteranceAppender(new_utterances)
        updated = module.visit(transformer)
        
        self.assertTrue(transformer.utterances_updated)
        self.assertIn('"first utterance"', updated.code)
        self.assertIn('"second utterance"', updated.code)


class TestStateExtractor(unittest.TestCase):
    """Test the StateExtractor visitor."""
    
    def test_extract_complete_state(self):
        """Test extracting complete state from a command file."""
        code = '''class Signature:
    """Command docstring"""
    
    class Input(BaseModel):
        name: str = Field(description="Name field")
        age: int
        email: str = Field(description="Email", examples=["test@example.com"])
    
    class Output(BaseModel):
        result: str
        status: bool = Field(description="Success status")
    
    plain_utterances = [
        "create user",
        "add person"
    ]'''
        
        module = cst.parse_module(code)
        extractor = StateExtractor()
        # Use wrapper to walk the tree
        wrapper = cst.MetadataWrapper(module)
        wrapper.visit(extractor)
        
        # Check docstring detection
        self.assertTrue(extractor.has_signature_docstring)
        
        # Check input fields
        self.assertEqual(len(extractor.input_fields), 3)
        
        name_field = next(f for f in extractor.input_fields if f['name'] == 'name')
        self.assertTrue(name_field['has_description'])
        self.assertFalse(name_field['has_examples'])
        
        age_field = next(f for f in extractor.input_fields if f['name'] == 'age')
        self.assertFalse(age_field['has_description'])
        self.assertFalse(age_field['has_examples'])
        
        email_field = next(f for f in extractor.input_fields if f['name'] == 'email')
        self.assertTrue(email_field['has_description'])
        self.assertTrue(email_field['has_examples'])
        
        # Check output fields
        self.assertEqual(len(extractor.output_fields), 2)
        
        result_field = next(f for f in extractor.output_fields if f['name'] == 'result')
        self.assertFalse(result_field['has_description'])
        
        status_field = next(f for f in extractor.output_fields if f['name'] == 'status')
        self.assertTrue(status_field['has_description'])
        
        # Check utterances
        self.assertEqual(len(extractor.plain_utterances), 2)
        self.assertIn("create user", extractor.plain_utterances)
        self.assertIn("add person", extractor.plain_utterances)
    
    def test_extract_minimal_state(self):
        """Test extracting state from minimal command file."""
        code = '''class Signature:
    class Input(BaseModel):
        value: str
    
    class Output(BaseModel):
        result: str
    
    plain_utterances = []'''
        
        module = cst.parse_module(code)
        extractor = StateExtractor()
        # Use wrapper to walk the tree
        wrapper = cst.MetadataWrapper(module)
        wrapper.visit(extractor)
        
        self.assertFalse(extractor.has_signature_docstring)
        self.assertEqual(len(extractor.input_fields), 1)
        self.assertEqual(len(extractor.output_fields), 1)
        self.assertEqual(len(extractor.plain_utterances), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for multiple transformers."""
    
    def test_full_transformation_pipeline(self):
        """Test applying all transformers in sequence."""
        code = '''from pydantic import BaseModel, Field

class Signature:
    class Input(BaseModel):
        name: str
        age: int = Field(description="Age in years")
    
    class Output(BaseModel):
        message: str
    
    plain_utterances = [
        "existing utterance"
    ]'''
        
        module = cst.parse_module(code)
        
        # Apply transformers in sequence
        
        # 1. Add docstring
        docstring_updater = SignatureDocstringUpdater("Process user information")
        module = module.visit(docstring_updater)
        
        # 2. Add field metadata
        field_metadata = {
            "Input.name": {"description": "User's full name", "examples": ["John Doe"]},
            "Input.age": {"examples": [25, 30]},  # Should only add examples
            "Output.message": {"description": "Response message"}
        }
        field_updater = FieldMetadataUpdater(field_metadata)
        module = module.visit(field_updater)
        
        # 3. Append utterances
        utterance_appender = UtteranceAppender(["new utterance", "another one"])
        module = module.visit(utterance_appender)
        
        result = module.code
        
        # Verify all updates were applied
        self.assertIn('"""Process user information"""', result)
        self.assertIn('"User\'s full name"', result)  # Description added
        self.assertIn('description="Age in years"', result)  # Preserved
        self.assertIn('[25, 30]', result)  # Examples added
        self.assertIn('"existing utterance"', result)  # Preserved
        self.assertIn('"new utterance"', result)  # Added
        self.assertIn('"another one"', result)  # Added
    
    def test_idempotent_transformations(self):
        """Test that running transformers multiple times is idempotent."""
        code = '''class Signature:
    """Existing docstring"""
    
    class Input(BaseModel):
        name: str = Field(description="Name", examples=["John"])
    
    plain_utterances = ["utterance1"]'''
        
        module = cst.parse_module(code)
        
        # Run transformers twice
        for _ in range(2):
            module = module.visit(SignatureDocstringUpdater("New docstring"))
            module = module.visit(FieldMetadataUpdater({
                "Input.name": {"description": "New desc", "examples": ["Jane"]}
            }))
            module = module.visit(UtteranceAppender(["utterance1", "utterance2"]))
        
        result = module.code
        
        # Should still have original content (not duplicated)
        self.assertEqual(result.count('"""Existing docstring"""'), 1)
        self.assertEqual(result.count('description="Name"'), 1)
        self.assertEqual(result.count('"utterance1"'), 1)
        self.assertEqual(result.count('"utterance2"'), 1)


if __name__ == '__main__':
    unittest.main()
