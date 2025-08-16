"""Test that the LibCST transformers handle Annotated fields correctly."""

import libcst as cst
from fastworkflow.build.libcst_transformers import (
    FieldMetadataUpdater,
    StateExtractor
)


def test_annotated_field_not_duplicated():
    """Test that fields using Annotated[type, Field(...)] don't get duplicate Field assignments."""
    
    code = '''from typing import Annotated
from pydantic import BaseModel, Field

class Signature:
    class Input(BaseModel):
        order_id: Annotated[
            str,
            Field(
                default="NOT_FOUND",
                description="The order ID to cancel (must start with #)",
                pattern=r"^(#[\w\d]+|NOT_FOUND)$",
                examples=["#123", "#abc123", "#order456"]
            )
        ]
        
        regular_field: str'''
    
    module = cst.parse_module(code)
    
    # Try to add metadata
    metadata = {
        "Input.order_id": {
            "description": "New description",
            "examples": ["new1", "new2"]
        },
        "Input.regular_field": {
            "description": "Regular field description",
            "examples": ["ex1", "ex2"]
        }
    }
    
    transformer = FieldMetadataUpdater(metadata)
    updated = module.visit(transformer)
    
    result = updated.code
    
    # Check that order_id doesn't have a duplicate Field assignment
    assert '] = Field(' not in result, "Annotated field should not get duplicate Field assignment"
    
    # Check that regular_field does get Field assignment
    assert 'regular_field: str = Field(' in result, "Regular field should get Field assignment"
    
    print("✅ Annotated field test passed!")


def test_state_extractor_with_annotated():
    """Test that StateExtractor correctly identifies metadata in Annotated fields."""
    
    code = '''from typing import Annotated
from pydantic import BaseModel, Field

class Signature:
    class Input(BaseModel):
        order_id: Annotated[
            str,
            Field(
                description="Order ID",
                examples=["#123"]
            )
        ]
        
        name: str = Field(description="Name")
        age: int'''
    
    module = cst.parse_module(code)
    extractor = StateExtractor()
    wrapper = cst.MetadataWrapper(module)
    wrapper.visit(extractor)
    
    # Find the fields
    order_field = next(f for f in extractor.input_fields if f['name'] == 'order_id')
    name_field = next(f for f in extractor.input_fields if f['name'] == 'name')
    age_field = next(f for f in extractor.input_fields if f['name'] == 'age')
    
    # Check that metadata is correctly detected
    assert order_field['has_description'] == True, "Should detect description in Annotated"
    assert order_field['has_examples'] == True, "Should detect examples in Annotated"
    
    assert name_field['has_description'] == True, "Should detect description in regular Field"
    
    assert age_field['has_description'] == False, "Should not have description"
    
    print("✅ State extractor with Annotated test passed!")


if __name__ == "__main__":
    test_annotated_field_not_duplicated()
    test_state_extractor_with_annotated()
    print("\nAll Annotated field tests passed! 🎉")
