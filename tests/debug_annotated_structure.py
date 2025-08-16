"""Debug script to understand the AST structure of Annotated fields."""

import libcst as cst


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
        ]'''

module = cst.parse_module(code)

# Find the AnnAssign node using a visitor
class DebugVisitor(cst.CSTVisitor):
    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if isinstance(node.target, cst.Name) and node.target.value == "order_id":
            print("Found order_id field")
            print(f"Annotation type: {type(node.annotation)}")
            print(f"Annotation.annotation type: {type(node.annotation.annotation)}")
            
            if isinstance(node.annotation.annotation, cst.Subscript):
                subscript = node.annotation.annotation
                print(f"Subscript.value: {subscript.value}")
                print(f"Subscript.slice type: {type(subscript.slice)}")
                
                # The slice contains the elements
                for i, element in enumerate(subscript.slice):
                    print(f"\nSlice element {i}:")
                    print(f"  Type: {type(element)}")
                    if isinstance(element, cst.SubscriptElement):
                        print(f"  Slice type: {type(element.slice)}")
                        if isinstance(element.slice, cst.Index):
                            print(f"  Index.value type: {type(element.slice.value)}")
                            print(f"  Index.value: {element.slice.value}")
                            
                            # Check if it's a Field call
                            if isinstance(element.slice.value, cst.Call):
                                call = element.slice.value
                                if isinstance(call.func, cst.Name):
                                    print(f"  Call func name: {call.func.value}")
                                    
                                    # Check arguments
                                    for arg in call.args:
                                        if arg.keyword:
                                            print(f"    Arg: {arg.keyword.value}")

visitor = DebugVisitor()
wrapper = cst.MetadataWrapper(module)
wrapper.visit(visitor)
