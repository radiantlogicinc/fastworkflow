"""LibCST transformers for targeted command file updates - Fixed for Annotated fields.

This module provides transformer classes that make surgical, targeted updates
to command files while preserving formatting, comments, and existing content.
"""

import libcst as cst
from typing import Optional, List, Dict, Any, Set, Union
import logging

logger = logging.getLogger(__name__)


class SignatureDocstringUpdater(cst.CSTTransformer):
    """Updates Signature class docstring only if missing or empty."""
    
    def __init__(self, new_docstring: str):
        self.new_docstring = new_docstring
        self.in_signature_class = False
        self.signature_updated = False
    
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Track when we enter the Signature class."""
        if node.name.value == "Signature":
            self.in_signature_class = True
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """Update Signature class docstring if needed."""
        if original_node.name.value == "Signature":
            self.in_signature_class = False
            
            # Check if docstring exists and is non-empty
            existing_docstring = self._get_docstring(updated_node)
            if not existing_docstring or existing_docstring.strip() == '':
                # Add or update docstring
                logger.debug(f"Adding docstring to Signature class")
                self.signature_updated = True
                return self._set_docstring(updated_node, self.new_docstring)
            else:
                logger.debug(f"Signature class already has docstring: {existing_docstring[:50]}...")
        
        return updated_node
    
    def _get_docstring(self, node: cst.ClassDef) -> Optional[str]:
        """Extract docstring from class if it exists."""
        if node.body and node.body.body:
            first_stmt = node.body.body[0]
            if isinstance(first_stmt, cst.SimpleStatementLine):
                if first_stmt.body and isinstance(first_stmt.body[0], cst.Expr):
                    expr_value = first_stmt.body[0].value
                    if isinstance(expr_value, (cst.SimpleString, cst.ConcatenatedString)):
                        # Extract the actual string value
                        if isinstance(expr_value, cst.SimpleString):
                            return expr_value.value.strip('"""\'\'\'')
                        return None
        return None
    
    def _set_docstring(self, node: cst.ClassDef, docstring: str) -> cst.ClassDef:
        """Add or update docstring in class."""
        # Create docstring node
        docstring_node = cst.SimpleStatementLine(
            body=[cst.Expr(cst.SimpleString(f'"""{docstring}"""'))]
        )
        
        # Check if first statement is already a docstring
        if node.body and node.body.body:
            first_stmt = node.body.body[0]
            if isinstance(first_stmt, cst.SimpleStatementLine):
                if first_stmt.body and isinstance(first_stmt.body[0], cst.Expr):
                    if isinstance(first_stmt.body[0].value, (cst.SimpleString, cst.ConcatenatedString)):
                        # Replace existing docstring
                        new_body = [docstring_node] + list(node.body.body[1:])
                        return node.with_changes(
                            body=node.body.with_changes(body=new_body)
                        )
        
        # Insert docstring at the beginning
        new_body = [docstring_node] + list(node.body.body)
        return node.with_changes(
            body=node.body.with_changes(body=new_body)
        )


class FieldMetadataUpdater(cst.CSTTransformer):
    """Updates Input/Output field metadata only if missing - handles Annotated types."""
    
    def __init__(self, field_metadata: Dict[str, Dict[str, Any]]):
        """
        Args:
            field_metadata: Dict with keys like "Input.field_name" or "Output.field_name"
                          and values containing 'description', 'examples', 'pattern'
        """
        self.field_metadata = field_metadata
        self.current_class = None
        self.in_signature = False
        self.fields_updated = []
    
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Track class hierarchy."""
        if node.name.value == "Signature":
            self.in_signature = True
        elif self.in_signature and node.name.value in ("Input", "Output"):
            self.current_class = node.name.value
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """Reset state when leaving classes."""
        if original_node.name.value == "Signature":
            self.in_signature = False
        elif original_node.name.value in ("Input", "Output"):
            self.current_class = None
        
        return updated_node
    
    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign) -> cst.AnnAssign:
        """Update field assignments with metadata - handles both regular and Annotated fields."""
        if not self.current_class or not isinstance(updated_node.target, cst.Name):
            return updated_node
        
        field_name = updated_node.target.value
        field_key = f"{self.current_class}.{field_name}"
        
        if field_key not in self.field_metadata:
            return updated_node
        
        metadata = self.field_metadata[field_key]
        
        # Check if this is an Annotated type with Field() inside
        if self._has_annotated_field(updated_node):
            # Field is inside Annotated, DO NOT add assignment
            logger.debug(f"Field {field_key} uses Annotated with Field inside, skipping assignment")
            # TODO: In future, we could update the Field inside Annotated, but that's complex
            return updated_node
        
        # Check if Field() call already exists as assignment
        if updated_node.value and self._is_field_call(updated_node.value):
            # Merge metadata into existing Field() call
            result = self._merge_field_metadata(updated_node, metadata)
            if result != updated_node:
                self.fields_updated.append(field_key)
                logger.debug(f"Updated existing Field() for {field_key}")
            return result
        elif not updated_node.value:
            # No assignment value, add Field() call
            result = self._add_field_call(updated_node, metadata)
            self.fields_updated.append(field_key)
            logger.debug(f"Added new Field() for {field_key}")
            return result
        else:
            # Has a value but it's not Field(), don't modify
            logger.debug(f"Field {field_key} has non-Field value, skipping")
            return updated_node
    
    def _has_annotated_field(self, node: cst.AnnAssign) -> bool:
        """Check if the annotation uses Annotated[type, Field(...)]."""
        if not node.annotation:
            return False
        
        annotation = node.annotation.annotation
        
        # Check if it's a Subscript (like Annotated[...])
        if isinstance(annotation, cst.Subscript):
            # Check if the value is Name with value "Annotated"
            if isinstance(annotation.value, cst.Name) and annotation.value.value == "Annotated":
                # Check if there are slice elements (the [...] part)
                if isinstance(annotation.slice, (list, tuple)):
                    slice_elements = annotation.slice
                else:
                    slice_elements = [annotation.slice]
                
                # Look for Field() in the slice elements
                for element in slice_elements:
                    if isinstance(element, cst.SubscriptElement):
                        if isinstance(element.slice, cst.Index):
                            if self._contains_field_call(element.slice.value):
                                return True
        
        return False
    
    def _contains_field_call(self, node: cst.BaseExpression) -> bool:
        """Recursively check if a node contains a Field() call."""
        if isinstance(node, cst.Call):
            if isinstance(node.func, cst.Name) and node.func.value == "Field":
                return True
        
        # Check in tuples/lists
        if isinstance(node, cst.Tuple):
            for element in node.elements:
                if isinstance(element, cst.Element):
                    if self._contains_field_call(element.value):
                        return True
        
        return False
    
    def _is_field_call(self, node: cst.BaseExpression) -> bool:
        """Check if node is a Field() call."""
        return (isinstance(node, cst.Call) and 
                isinstance(node.func, cst.Name) and 
                node.func.value == "Field")
    
    def _get_existing_field_args(self, call: cst.Call) -> Dict[str, cst.Arg]:
        """Extract existing keyword arguments from Field() call."""
        existing_args = {}
        for arg in call.args:
            if arg.keyword:
                existing_args[arg.keyword.value] = arg
        return existing_args
    
    def _merge_field_metadata(self, node: cst.AnnAssign, metadata: Dict[str, Any]) -> cst.AnnAssign:
        """Merge new metadata into existing Field() call, preserving existing values."""
        if not isinstance(node.value, cst.Call):
            return node
        
        existing_args = self._get_existing_field_args(node.value)
        new_args = []
        
        # Keep all existing arguments
        for arg in node.value.args:
            new_args.append(arg)
        
        # Only add missing metadata
        if "description" not in existing_args and metadata.get("description"):
            new_args.append(
                cst.Arg(
                    keyword=cst.Name("description"),
                    value=cst.SimpleString(f'"{metadata["description"]}"')
                )
            )
        
        if "examples" not in existing_args and metadata.get("examples"):
            # Create list elements with proper handling for different types
            example_elements = []
            for ex in metadata["examples"]:
                if isinstance(ex, str):
                    example_elements.append(cst.Element(cst.SimpleString(f'"{ex}"')))
                else:
                    # For non-string values, use Integer nodes
                    example_elements.append(cst.Element(cst.Integer(str(ex))))
            examples_list = cst.List(example_elements)
            new_args.append(
                cst.Arg(
                    keyword=cst.Name("examples"),
                    value=examples_list
                )
            )
        
        if "pattern" not in existing_args and metadata.get("pattern"):
            new_args.append(
                cst.Arg(
                    keyword=cst.Name("pattern"),
                    value=cst.SimpleString(f'"{metadata["pattern"]}"')
                )
            )
        
        # Only update if we actually added new arguments
        if len(new_args) > len(node.value.args):
            new_call = node.value.with_changes(args=new_args)
            return node.with_changes(value=new_call)
        
        return node
    
    def _add_field_call(self, node: cst.AnnAssign, metadata: Dict[str, Any]) -> cst.AnnAssign:
        """Add Field() call with metadata to a field that doesn't have one."""
        args = []
        
        if metadata.get("description"):
            args.append(
                cst.Arg(
                    keyword=cst.Name("description"),
                    value=cst.SimpleString(f'"{metadata["description"]}"')
                )
            )
        
        if metadata.get("examples"):
            # Create list elements with proper handling for different types
            example_elements = []
            for ex in metadata["examples"]:
                if isinstance(ex, str):
                    example_elements.append(cst.Element(cst.SimpleString(f'"{ex}"')))
                else:
                    # For non-string values, use Integer nodes
                    example_elements.append(cst.Element(cst.Integer(str(ex))))
            examples_list = cst.List(example_elements)
            args.append(
                cst.Arg(
                    keyword=cst.Name("examples"),
                    value=examples_list
                )
            )
        
        if metadata.get("pattern"):
            args.append(
                cst.Arg(
                    keyword=cst.Name("pattern"),
                    value=cst.SimpleString(f'"{metadata["pattern"]}"')
                )
            )
        
        field_call = cst.Call(
            func=cst.Name("Field"),
            args=args
        )
        
        return node.with_changes(value=field_call)


class UtteranceAppender(cst.CSTTransformer):
    """Appends new utterances to plain_utterances list without removing existing ones."""
    
    def __init__(self, new_utterances: List[str]):
        self.new_utterances = new_utterances
        self.in_signature = False
        self.utterances_updated = False
        self.existing_utterances: Set[str] = set()
    
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Track when we're in the Signature class."""
        if node.name.value == "Signature":
            self.in_signature = True
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """Reset state when leaving Signature class."""
        if original_node.name.value == "Signature":
            self.in_signature = False
        return updated_node
    
    def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
        """Update plain_utterances assignment."""
        if not self.in_signature:
            return updated_node
        
        # Check if this is plain_utterances assignment
        for target in updated_node.targets:
            if isinstance(target.target, cst.Name) and target.target.value == "plain_utterances":
                return self._append_utterances(updated_node)
        
        return updated_node
    
    def _append_utterances(self, node: cst.Assign) -> cst.Assign:
        """Append new utterances to existing list."""
        if not isinstance(node.value, cst.List):
            logger.warning("plain_utterances is not a list, skipping update")
            return node
        
        # Extract existing utterances
        existing_utterances = set()
        existing_elements = []
        
        for element in node.value.elements:
            if isinstance(element, cst.Element):
                if isinstance(element.value, cst.SimpleString):
                    # Parse the string value
                    utterance = self._extract_string_value(element.value)
                    existing_utterances.add(utterance)
                    existing_elements.append(element)
                else:
                    # Preserve non-string elements as-is
                    existing_elements.append(element)
        
        self.existing_utterances = existing_utterances
        
        # Only add truly new utterances
        new_elements = existing_elements.copy()
        added_count = 0
        
        for utterance in self.new_utterances:
            if utterance not in existing_utterances:
                # Determine quote style based on existing elements
                quote_style = self._get_quote_style(node.value)
                new_elements.append(
                    cst.Element(cst.SimpleString(f'{quote_style}{utterance}{quote_style}'))
                )
                added_count += 1
                logger.debug(f"Adding new utterance: {utterance}")
        
        if added_count > 0:
            self.utterances_updated = True
            # Create new list with combined utterances
            new_list = cst.List(elements=new_elements)
            return node.with_changes(value=new_list)
        else:
            logger.debug("No new utterances to add")
        
        return node
    
    def _extract_string_value(self, node: cst.SimpleString) -> str:
        """Extract string value from SimpleString node."""
        value = node.value
        # Remove surrounding quotes
        if value.startswith('"""') or value.startswith("'''"):
            return value[3:-3]
        elif value.startswith('"') or value.startswith("'"):
            return value[1:-1]
        return value
    
    def _get_quote_style(self, list_node: cst.List) -> str:
        """Determine quote style from existing list elements."""
        for element in list_node.elements:
            if isinstance(element, cst.Element) and isinstance(element.value, cst.SimpleString):
                value = element.value.value
                if value.startswith('"'):
                    return '"'
                elif value.startswith("'"):
                    return "'"
        # Default to double quotes
        return '"'


class StateExtractor(cst.CSTVisitor):
    """Extract current state from a command file for analysis - handles Annotated fields."""
    
    def __init__(self):
        self.has_signature_docstring = False
        self.input_fields: List[Dict[str, Any]] = []
        self.output_fields: List[Dict[str, Any]] = []
        self.plain_utterances: List[str] = []
        self.current_class = None
        self.in_signature = False
    
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Track class hierarchy and extract docstrings."""
        if node.name.value == "Signature":
            self.in_signature = True
            # Check for docstring
            if node.body and node.body.body:
                first_stmt = node.body.body[0]
                if isinstance(first_stmt, cst.SimpleStatementLine):
                    if first_stmt.body and isinstance(first_stmt.body[0], cst.Expr):
                        if isinstance(first_stmt.body[0].value, (cst.SimpleString, cst.ConcatenatedString)):
                            docstring = self._extract_docstring(first_stmt.body[0].value)
                            if docstring and docstring.strip():
                                self.has_signature_docstring = True
        
        elif self.in_signature and node.name.value in ("Input", "Output"):
            self.current_class = node.name.value
    
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        """Reset state when leaving classes."""
        if original_node.name.value == "Signature":
            self.in_signature = False
        elif original_node.name.value in ("Input", "Output"):
            self.current_class = None
    
    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        """Extract field information - handles both regular and Annotated fields."""
        if self.current_class and isinstance(node.target, cst.Name):
            field_info = {
                'name': node.target.value,
                'type': self._extract_type(node.annotation) if node.annotation else "Any",
                'has_description': False,
                'has_examples': False,
                'has_pattern': False
            }
            
            # Check for Field() in Annotated type
            if self._check_annotated_field_metadata(node, field_info):
                # Metadata found in Annotated
                pass
            # Check if Field() call exists as assignment
            elif node.value and isinstance(node.value, cst.Call):
                if isinstance(node.value.func, cst.Name) and node.value.func.value == "Field":
                    for arg in node.value.args:
                        if arg.keyword:
                            if arg.keyword.value == "description":
                                field_info['has_description'] = True
                            elif arg.keyword.value == "examples":
                                field_info['has_examples'] = True
                            elif arg.keyword.value == "pattern":
                                field_info['has_pattern'] = True
            
            if self.current_class == "Input":
                self.input_fields.append(field_info)
            elif self.current_class == "Output":
                self.output_fields.append(field_info)
    
    def _check_annotated_field_metadata(self, node: cst.AnnAssign, field_info: Dict[str, Any]) -> bool:
        """Check if field uses Annotated[type, Field(...)] and extract metadata info."""
        if not node.annotation:
            return False
        
        annotation = node.annotation.annotation
        
        # Check if it's a Subscript (like Annotated[...])
        if isinstance(annotation, cst.Subscript):
            # Check if the value is Name with value "Annotated"
            if isinstance(annotation.value, cst.Name) and annotation.value.value == "Annotated":
                # Check slice elements for Field()
                if isinstance(annotation.slice, (list, tuple)):
                    slice_elements = annotation.slice
                else:
                    slice_elements = [annotation.slice]
                
                for element in slice_elements:
                    if isinstance(element, cst.SubscriptElement):
                        if isinstance(element.slice, cst.Index):
                            # Check if this element is a Field() call
                            if isinstance(element.slice.value, cst.Call):
                                if isinstance(element.slice.value.func, cst.Name) and element.slice.value.func.value == "Field":
                                    self._extract_field_metadata(element.slice.value, field_info)
                                    return True
        
        return False
    
    def _extract_field_metadata(self, node: cst.BaseExpression, field_info: Dict[str, Any]) -> None:
        """Extract Field() metadata from a node."""
        if isinstance(node, cst.Call):
            if isinstance(node.func, cst.Name) and node.func.value == "Field":
                for arg in node.args:
                    if arg.keyword:
                        if arg.keyword.value == "description":
                            field_info['has_description'] = True
                        elif arg.keyword.value == "examples":
                            field_info['has_examples'] = True
                        elif arg.keyword.value == "pattern":
                            field_info['has_pattern'] = True
        
        # Check in tuples
        if isinstance(node, cst.Tuple):
            for element in node.elements:
                if isinstance(element, cst.Element):
                    self._extract_field_metadata(element.value, field_info)
    
    def visit_Assign(self, node: cst.Assign) -> None:
        """Extract plain_utterances."""
        if self.in_signature:
            for target in node.targets:
                if isinstance(target.target, cst.Name) and target.target.value == "plain_utterances":
                    if isinstance(node.value, cst.List):
                        for element in node.value.elements:
                            if isinstance(element, cst.Element) and isinstance(element.value, cst.SimpleString):
                                utterance = self._extract_string_value(element.value)
                                self.plain_utterances.append(utterance)
    
    def _extract_docstring(self, node: Union[cst.SimpleString, cst.ConcatenatedString]) -> Optional[str]:
        """Extract docstring value."""
        if isinstance(node, cst.SimpleString):
            value = node.value
            if value.startswith('"""') or value.startswith("'''"):
                return value[3:-3]
            elif value.startswith('"') or value.startswith("'"):
                return value[1:-1]
        return None
    
    def _extract_type(self, annotation: cst.Annotation) -> str:
        """Extract type annotation as string."""
        if annotation and annotation.annotation:
            # Use LibCST's code generation to get the type as string
            return cst.Module([]).code_for_node(annotation.annotation)
        return "Any"
    
    def _extract_string_value(self, node: cst.SimpleString) -> str:
        """Extract string value from SimpleString node."""
        value = node.value
        if value.startswith('"""') or value.startswith("'''"):
            return value[3:-3]
        elif value.startswith('"') or value.startswith("'"):
            return value[1:-1]
        return value
