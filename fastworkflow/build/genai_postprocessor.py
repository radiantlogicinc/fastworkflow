"""GenAI Post-Processing Component for FastWorkflow Build Tool.

This module uses DSPy to enhance generated command files with AI-generated content including:
- Field descriptions, examples, and constraints
- Natural language utterances
- Dynamic docstrings
- Workflow descriptions

The implementation uses LibCST for targeted, source-preserving updates that only modify
what's necessary while preserving all existing content, formatting, and comments.
"""

import os
import logging
from typing import Dict, List, Any, Optional
import traceback

import libcst as cst
import dspy
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.utils import dspy_utils
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, FunctionInfo
from fastworkflow.build.libcst_transformers import (
    SignatureDocstringUpdater,
    FieldMetadataUpdater,
    UtteranceAppender,
    StateExtractor
)
from fastworkflow.utils.logging import logger

# ============================================================================
# DSPy Signatures
# ============================================================================

class FieldMetadataSignature(dspy.Signature):
    """Generate metadata for a command field."""
    
    field_name: str = dspy.InputField(desc="Name of the field")
    field_type: str = dspy.InputField(desc="Type annotation of the field")
    method_docstring: str = dspy.InputField(desc="Docstring of the method this field belongs to")
    method_name: str = dspy.InputField(desc="Name of the method")
    context_name: str = dspy.InputField(desc="Name of the context/class")
    is_input: bool = dspy.InputField(desc="Whether this is an input field (True) or output field (False)")
    
    description: str = dspy.OutputField(desc="Clear, concise description of the field (1-2 sentences)")
    examples: List[str] = dspy.OutputField(desc="2-3 example values for the field")
    pattern: str = dspy.OutputField(desc="Regular expression pattern for field validation (e.g., '^[A-Z][a-z]+$' for capitalized words, '^\\d{3}-\\d{3}-\\d{4}$' for phone numbers). Return empty string if no pattern constraint is needed.")


class UtteranceGeneratorSignature(dspy.Signature):
    """Generate minimal natural language utterances for a command."""
    
    command_name: str = dspy.InputField(desc="Name of the command")
    command_docstring: str = dspy.InputField(desc="Docstring describing what the command does")
    command_input_fields: str = dspy.InputField(desc="JSON string of input fields with name, type, and description")
    
    utterances: List[str] = dspy.OutputField(
        desc="Minimal list of natural language utterances covering all parameter combinations. "
             "Include utterances with no parameters, single parameters, and all parameters. "
             "Vary the parameter values across utterances. Keep utterances natural and concise."
    )


class SignatureDocstringSignature(dspy.Signature):
    """Generate docstring for Command Signature class."""
    
    command_name: str = dspy.InputField(desc="Name of the command")
    input_fields_json: str = dspy.InputField(desc="JSON string of input fields with name, type, and description")
    output_fields_json: str = dspy.InputField(desc="JSON string of output fields with name, type, and description")
    context_name: str = dspy.InputField(desc="Name of the context/class this command belongs to")
    
    docstring: str = dspy.OutputField(
        desc="Comprehensive docstring for the command optimized for an LLM agent. "
             "Include a brief description of the command "
             "with a parameters section that clearly explains the inputs and outputs (if inputs and outputs exist), "
             "and optionally an examples section illustrating the command in xml format as follows: "
             "<command_name><input_parameter_name>param 1 value</input_parameter_name><input_parameter_name>param 2 value</input_parameter_name></command_name>. "
             "Replace 'command_name' with the actual command name. "
             "Replace 'input_parameter_name' with the actual name of the input parameter. "
             "Replace 'param 1 value' and 'param 2 value' with the actual values of the input parameters. "
    )


class ContextDocstringSignature(dspy.Signature):
    """Generate docstring for context handler file."""
    
    context_name: str = dspy.InputField(desc="Name of the context")
    commands_json: str = dspy.InputField(desc="JSON string of commands with name and docstring")
    
    docstring: str = dspy.OutputField(
        desc="Aggregated docstring summarizing the context and its commands. "
             "Should provide an overview of what the context handles and list available commands."
    )


class WorkflowDescriptionSignature(dspy.Signature):
    """Generate overall workflow description."""
    
    contexts_json: str = dspy.InputField(
        desc="JSON string of contexts with context_name, docstring, and commands"
    )
    global_commands_json: str = dspy.InputField(
        desc="JSON string of global commands with name and docstring"
    )
    
    description: str = dspy.OutputField(
        desc="High-level workflow overview describing the purpose, capabilities, "
             "and structure of the workflow. Should be comprehensive yet readable."
    )


# ============================================================================
# DSPy Modules
# ============================================================================

class FieldMetadataGenerator(dspy.Module):
    """DSPy module for generating field metadata."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(FieldMetadataSignature)
    
    def forward(self, field_name, field_type, method_docstring, method_name, context_name, is_input):
        """Generate metadata for a field."""
        try:
            return self.generate(
                field_name=field_name,
                field_type=field_type,
                method_docstring=method_docstring or "",
                method_name=method_name,
                context_name=context_name,
                is_input=is_input,
            )
        except Exception as e:
            logger.warning(f"Failed to generate metadata for field {field_name}: {e}")
            # Return defaults on failure
            return type('Result', (), {
                'description': f"The {field_name} parameter",
                'examples': [],
                'pattern': ""
            })()


class UtteranceGenerator(dspy.Module):
    """DSPy module for generating command utterances."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(UtteranceGeneratorSignature)
    
    def forward(self, command_name, command_docstring, input_fields):
        """Generate utterances for a command."""
        try:
            # Convert input_fields list to JSON string for DSPy
            import json
            input_fields_json = json.dumps(input_fields) if input_fields else "[]"
            
            result = self.generate(
                command_name=command_name,
                command_docstring=command_docstring or "",
                command_input_fields=input_fields_json
            )
            # Ensure we return a list of strings
            if hasattr(result, 'utterances') and isinstance(result.utterances, list):
                return result.utterances
            return [command_name.lower().replace('_', ' ')]
        except Exception as e:
            logger.warning(f"Failed to generate utterances for command {command_name}: {e}")
            # Return default utterance on failure
            return [command_name.lower().replace('_', ' ')]


class DocstringGenerator(dspy.Module):
    """DSPy module for generating docstrings."""
    
    def __init__(self):
        super().__init__()
        self.signature_docstring = dspy.ChainOfThought(SignatureDocstringSignature)
        self.context_docstring = dspy.ChainOfThought(ContextDocstringSignature)
    
    def generate_signature_docstring(self, command_name, input_fields, output_fields, context_name):
        """Generate docstring for a command signature."""
        try:
            import json
            result = self.signature_docstring(
                command_name=command_name,
                input_fields_json=json.dumps(input_fields) if input_fields else "[]",
                output_fields_json=json.dumps(output_fields) if output_fields else "[]",
                context_name=context_name,
            )
            return result.docstring.strip('"""')
        except Exception as e:
            logger.warning(f"Failed to generate signature docstring for {command_name}: {e}")
            return f"Execute {command_name} command."
    
    def generate_context_docstring(self, context_name, commands):
        """Generate docstring for a context handler."""
        try:
            import json
            result = self.context_docstring(
                context_name=context_name,
                commands_json=json.dumps(commands) if commands else "[]"
            )
            return result.docstring
        except Exception as e:
            logger.warning(f"Failed to generate context docstring for {context_name}: {e}")
            return f"Context handler for {context_name}."


class WorkflowDescriptionGenerator(dspy.Module):
    """DSPy module for generating workflow descriptions."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(WorkflowDescriptionSignature)
    
    def forward(self, contexts, global_commands):
        """Generate workflow description."""
        try:
            import json
            result = self.generate(
                contexts_json=json.dumps(contexts) if contexts else "{}",
                global_commands_json=json.dumps(global_commands) if global_commands else "[]"
            )
            return result.description
        except Exception as e:
            logger.warning(f"Failed to generate workflow description: {e}")
            return "FastWorkflow automated workflow system."


class GenAIPostProcessor:
    """Post-processor using LibCST for targeted, source-preserving updates."""
    
    def __init__(self):
        """Initialize the post-processor with DSPy configuration."""
        # Get model and API key from FastWorkflow environment
        self.lm = dspy_utils.get_lm("LLM_COMMAND_METADATA_GEN", "LITELLM_API_KEY_COMMANDMETADATA_GEN", max_tokens=2000)
        
        # Initialize DSPy modules (reuse from original)
        self.field_generator = FieldMetadataGenerator()
        self.utterance_generator = UtteranceGenerator()
        self.docstring_generator = DocstringGenerator()
        self.workflow_generator = WorkflowDescriptionGenerator()
        
        # Track statistics
        self.stats = {
            'files_processed': 0,
            'files_updated': 0,
            'docstrings_added': 0,
            'fields_updated': 0,
            'utterances_added': 0
        }
    
    def process_workflow(self, workflow_path: str, classes: Dict[str, ClassInfo], 
                        functions: Optional[Dict[str, FunctionInfo]] = None) -> bool:
        """Process all command files in the workflow with targeted updates.
        
        Args:
            workflow_path: Path to the workflow directory
            classes: Dictionary of ClassInfo objects from the build phase
            functions: Dictionary of FunctionInfo objects (optional)
        
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        try:
            logger.info("Starting targeted GenAI post-processing with LibCST...")

            # Process command files
            commands_dir = os.path.join(workflow_path, "_commands")
            if not os.path.exists(commands_dir):
                logger.error(f"Commands directory not found: {commands_dir}")
                return False

            # Track all contexts and commands for workflow description
            all_contexts = {}
            global_commands = []

            # Process context-specific commands
            for context_dir in os.listdir(commands_dir):
                context_path = os.path.join(commands_dir, context_dir)
                if os.path.isdir(context_path):
                    context_commands = self._process_context_commands(
                        context_path, context_dir, classes.get(context_dir)
                    )
                    if context_commands:
                        all_contexts[context_dir] = {
                            'context_name': context_dir,
                            'commands': context_commands,
                            'docstring': ""
                        }

                    # Generate context handler docstring (still uses original method)
                    self._generate_context_handler_docstring(context_path, context_dir, context_commands)

            # Process global commands
            for file_name in os.listdir(commands_dir):
                file_path = os.path.join(commands_dir, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.py') and not file_name.startswith('_'):
                    command_name = file_name[:-3]
                    func_info = functions.get(command_name) if functions else None
                    if self._process_command_file_targeted(file_path, command_name, None, func_info):
                        global_commands.append({
                            'name': command_name,
                            'docstring': func_info.docstring if func_info else ""
                        })

            # Generate workflow description (reuse original method)
            self._generate_workflow_description(workflow_path, all_contexts, global_commands)

            # Log statistics
            logger.info("Targeted post-processing completed:")
            logger.info(f"  Files processed: {self.stats['files_processed']}")
            logger.info(f"  Files updated: {self.stats['files_updated']}")
            logger.info(f"  Docstrings added: {self.stats['docstrings_added']}")
            logger.info(f"  Fields updated: {self.stats['fields_updated']}")
            logger.info(f"  Utterances added: {self.stats['utterances_added']}")

            return True

        except Exception as e:
            logger.error(f"Error during targeted GenAI post-processing: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _process_context_commands(self, context_path: str, context_name: str, 
                                 class_info: Optional[ClassInfo]) -> List[Dict[str, str]]:
        """Process all command files in a context directory."""
        commands = []
        
        for file_name in os.listdir(context_path):
            if file_name.endswith('.py') and not file_name.startswith('_'):
                file_path = os.path.join(context_path, file_name)
                command_name = file_name[:-3]
                
                # Find corresponding method info
                method_info = None
                if class_info:
                    for method in class_info.methods:
                        if method.name == command_name:
                            method_info = method
                            break
                
                if self._process_command_file_targeted(file_path, command_name, context_name, method_info):
                    commands.append({
                        'name': command_name,
                        'docstring': method_info.docstring if method_info else ""
                    })
        
        return commands
    
    def _process_command_file_targeted(self, file_path: str, command_name: str, 
                                      context_name: Optional[str], 
                                      method_or_func_info: Optional[Any]) -> bool:
        """Process a command file with targeted, laser-focused updates using LibCST.
        
        This method:
        1. Reads the file preserving formatting
        2. Extracts current state
        3. Generates only missing content
        4. Applies targeted transformations
        5. Writes back only if changes were made
        """
        try:
            logger.debug(f"Processing command file with targeted updates: {file_path}")
            self.stats['files_processed'] += 1
            
            # Read the original file content
            with open(file_path, 'r') as f:
                original_content = f.read()
            
            # Parse with LibCST to preserve formatting
            try:
                module = cst.parse_module(original_content)
            except Exception as e:
                logger.error(f"Failed to parse {file_path} with LibCST: {e}")
                return False
            
            # Extract current state
            current_state = self._extract_current_state(module)
            
            # Generate enhanced content only for missing elements
            enhanced_data = self._generate_enhanced_content(
                current_state, command_name, context_name, method_or_func_info
            )
            
            # Track if any changes were made
            changes_made = False
            
            # Apply targeted transformations
            
            # 1. Update Signature docstring (only if missing)
            if enhanced_data.get('signature_docstring'):
                transformer = SignatureDocstringUpdater(enhanced_data['signature_docstring'])
                module = module.visit(transformer)
                if transformer.signature_updated:
                    changes_made = True
                    self.stats['docstrings_added'] += 1
            
            # 2. Update field metadata (only add missing descriptions/examples)
            if enhanced_data.get('field_metadata'):
                transformer = FieldMetadataUpdater(enhanced_data['field_metadata'])
                module = module.visit(transformer)
                if transformer.fields_updated:
                    changes_made = True
                    self.stats['fields_updated'] += len(transformer.fields_updated)
            
            # 3. Append new utterances (never remove existing)
            if enhanced_data.get('new_utterances'):
                transformer = UtteranceAppender(enhanced_data['new_utterances'])
                module = module.visit(transformer)
                if transformer.utterances_updated:
                    changes_made = True
                    self.stats['utterances_added'] += len(enhanced_data['new_utterances']) - len(transformer.existing_utterances)
            
            # Write back only if changes were made
            if changes_made:
                updated_content = module.code
                with open(file_path, 'w') as f:
                    f.write(updated_content)
                logger.debug(f"Successfully applied targeted updates to {file_path}")
                self.stats['files_updated'] += 1
            else:
                logger.debug(f"No changes needed for {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing command file {file_path}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _extract_current_state(self, module: cst.Module) -> Dict[str, Any]:
        """Extract current state from LibCST module."""
        extractor = StateExtractor()
        # Use wrapper to walk the tree
        wrapper = cst.MetadataWrapper(module)
        wrapper.visit(extractor)
        
        return {
            'has_signature_docstring': extractor.has_signature_docstring,
            'input_fields': extractor.input_fields,
            'output_fields': extractor.output_fields,
            'plain_utterances': extractor.plain_utterances
        }
    
    def _generate_enhanced_content(self, current_state: Dict[str, Any], command_name: str,
                                  context_name: Optional[str], 
                                  method_or_func_info: Optional[Any]) -> Dict[str, Any]:
        """Generate enhanced content using DSPy modules, only for missing elements."""
        
        enhanced_data = {}

        # Only generate docstring if missing or empty
        if not current_state.get('has_signature_docstring'):
            try:
                with dspy.context(lm=self.lm):
                    signature_docstring = self.docstring_generator.generate_signature_docstring(
                        command_name=command_name,
                        input_fields=[{
                            'name': f['name'],
                            'type': f['type'],
                            'description': ''  # Will be generated if needed
                        } for f in current_state['input_fields']],
                        output_fields=[{
                            'name': f['name'],
                            'type': f['type'],
                            'description': ''
                        } for f in current_state['output_fields']],
                        context_name=context_name or "global"
                    )
                    enhanced_data['signature_docstring'] = signature_docstring
                    logger.debug(f"Generated new docstring for {command_name}")
            except Exception as e:
                logger.warning(f"Failed to generate docstring for {command_name}: {e}")

        # Generate field metadata only for fields missing descriptions/examples
        field_metadata = {}

        # Process input fields
        for field_info in current_state['input_fields']:
            if not field_info.get('has_description') or not field_info.get('has_examples'):
                try:
                    with dspy.context(lm=self.lm):
                        metadata = self.field_generator(
                            field_name=field_info['name'],
                            field_type=field_info['type'],
                            method_docstring=method_or_func_info.docstring if method_or_func_info else "",
                            method_name=command_name,
                            context_name=context_name or "global",
                            is_input=True
                        )

                        field_key = f"Input.{field_info['name']}"
                        field_metadata[field_key] = {}

                        if not field_info.get('has_description'):
                            field_metadata[field_key]['description'] = metadata.description

                        if not field_info.get('has_examples'):
                            field_metadata[field_key]['examples'] = metadata.examples

                        if metadata.pattern and metadata.pattern.strip():
                            field_metadata[field_key]['pattern'] = metadata.pattern

                        logger.debug(f"Generated metadata for {field_key}")
                except Exception as e:
                    logger.warning(f"Failed to generate metadata for Input.{field_info['name']}: {e}")

        # Process output fields
        for field_info in current_state['output_fields']:
            if not field_info.get('has_description') or not field_info.get('has_examples'):
                try:
                    with dspy.context(lm=self.lm):
                        metadata = self.field_generator(
                            field_name=field_info['name'],
                            field_type=field_info['type'],
                            method_docstring=method_or_func_info.docstring if method_or_func_info else "",
                            method_name=command_name,
                            context_name=context_name or "global",
                            is_input=False
                        )

                        field_key = f"Output.{field_info['name']}"
                        field_metadata[field_key] = {}

                        if not field_info.get('has_description'):
                            field_metadata[field_key]['description'] = metadata.description

                        if not field_info.get('has_examples'):
                            field_metadata[field_key]['examples'] = metadata.examples

                        logger.debug(f"Generated metadata for {field_key}")
                except Exception as e:
                    logger.warning(f"Failed to generate metadata for Output.{field_info['name']}: {e}")

        if field_metadata:
            enhanced_data['field_metadata'] = field_metadata

        # Generate new utterances to append
        try:
            with dspy.context(lm=self.lm):
                new_utterances = self.utterance_generator(
                    command_name=command_name,
                    command_docstring=method_or_func_info.docstring if method_or_func_info else "",
                    input_fields=[{
                        'name': f['name'],
                        'type': f['type'],
                        'description': ''
                    } for f in current_state['input_fields']]
                )

                # Filter out existing utterances
                existing_utterances = set(current_state.get('plain_utterances', []))
                if truly_new_utterances := [
                    u for u in new_utterances if u not in existing_utterances
                ]:
                    enhanced_data['new_utterances'] = truly_new_utterances
                    logger.debug(f"Generated {len(truly_new_utterances)} new utterances for {command_name}")
        except Exception as e:
            logger.warning(f"Failed to generate utterances for {command_name}: {e}")

        return enhanced_data
    
    def _generate_context_handler_docstring(self, context_path: str, context_name: str, 
                                           commands: List[Dict[str, str]]):
        """Generate and write docstring for _<Context>.py file using LibCST."""
        try:
            handler_file = os.path.join(context_path, f"_{context_name}.py")
            if not os.path.exists(handler_file):
                logger.debug(f"Context handler file not found: {handler_file}")
                return

            # Read the file
            with open(handler_file, 'r') as f:
                content = f.read()

            # Parse with LibCST
            module = cst.parse_module(content)

            # Generate context docstring
            with dspy.context(lm=self.lm):
                context_docstring = self.docstring_generator.generate_context_docstring(
                    context_name=context_name,
                    commands=commands
                )

            class UpdateContextDocstring(cst.CSTTransformer):
                def __init__(self, docstring: str):
                    self.docstring = docstring
                    self.updated = False

                def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
                    # Update the first class that matches the context name pattern
                    if (
                        self.updated
                        or updated_node.name.value != context_name
                        and not updated_node.name.value.endswith("Handler")
                    ):
                        return updated_node

                    self.updated = True
                    # Add or update docstring
                    if updated_node.body and updated_node.body.body:
                        first_stmt = updated_node.body.body[0]
                        if isinstance(first_stmt, cst.SimpleStatementLine) and (first_stmt.body and isinstance(first_stmt.body[0], cst.Expr)) and isinstance(first_stmt.body[0].value, (cst.SimpleString, cst.ConcatenatedString)):
                            new_docstring = cst.SimpleStatementLine(
                                body=[cst.Expr(cst.SimpleString(f'"""{self.docstring}"""'))]
                            )
                            new_body = [new_docstring] + list(updated_node.body.body[1:])
                            return updated_node.with_changes(
                                body=updated_node.body.with_changes(body=new_body)
                            )

                    # Add new docstring
                    docstring_node = cst.SimpleStatementLine(
                        body=[cst.Expr(cst.SimpleString(f'"""{self.docstring}"""'))]
                    )
                    new_body = [docstring_node] + list(updated_node.body.body)
                    return updated_node.with_changes(
                        body=updated_node.body.with_changes(body=new_body)
                    )


            transformer = UpdateContextDocstring(context_docstring)
            module = module.visit(transformer)

            if transformer.updated:
                # Write back
                with open(handler_file, 'w') as f:
                    f.write(module.code)
                logger.debug(f"Updated context handler docstring for {context_name}")

        except Exception as e:
            logger.error(f"Error generating context handler docstring: {e}")
            logger.error(traceback.format_exc())
    
    def _generate_workflow_description(self, workflow_path: str, 
                                      all_contexts: Dict[str, Dict],
                                      global_commands: List[Dict[str, str]]):
        """Generate workflow_description.txt file."""

        try:
            with dspy.context(lm=self.lm):
                description = self.workflow_generator(all_contexts, global_commands)
            
            desc_file = os.path.join(workflow_path, "workflow_description.txt")
            with open(desc_file, 'w') as f:
                f.write(description)
            
            logger.debug(f"Generated workflow description at {desc_file}")
            
        except Exception as e:
            logger.error(f"Error generating workflow description: {e}")
            logger.error(traceback.format_exc())


def run_genai_postprocessor(args, classes: Dict[str, ClassInfo], 
                          functions: Optional[Dict[str, FunctionInfo]] = None) -> bool:
    """Run the GenAI post-processor with targeted LibCST updates.
    
    This function uses LibCST to make surgical, targeted updates to command files
    while preserving all existing content, formatting, and comments.
    """
    try:
        # Initialize the post-processor
        processor = GenAIPostProcessor()
        
        # Process the workflow
        workflow_path = args.workflow_folderpath
        return processor.process_workflow(workflow_path, classes, functions)
        
    except Exception as e:
        logger.error(f"GenAI post-processing failed: {e}")
        logger.error(traceback.format_exc())
        # Don't fail the entire build if post-processing fails
        return True
