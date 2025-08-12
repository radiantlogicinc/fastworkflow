"""GenAI Post-Processing Component for FastWorkflow Build Tool.

This module uses DSPy to enhance generated command files with AI-generated content including:
- Field descriptions, examples, and constraints
- Natural language utterances
- Dynamic docstrings
- Workflow descriptions
"""

import os
import json
import ast
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import traceback

import dspy
from pydantic import BaseModel, Field
import fastworkflow

from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo, FunctionInfo
from fastworkflow.build.ast_class_extractor import parse_google_docstring
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
    original_docstring: str = dspy.InputField(desc="Original method/function docstring if available")
    
    docstring: str = dspy.OutputField(
        desc="Comprehensive docstring in Google style format. "
             "Include a brief description, Args section (if inputs exist), "
             "Returns section, and optionally Examples section."
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
    
    def generate_signature_docstring(self, command_name, input_fields, output_fields, context_name, original_docstring=""):
        """Generate docstring for a command signature."""
        try:
            import json
            result = self.signature_docstring(
                command_name=command_name,
                input_fields_json=json.dumps(input_fields) if input_fields else "[]",
                output_fields_json=json.dumps(output_fields) if output_fields else "[]",
                context_name=context_name,
                original_docstring=original_docstring
            )
            return result.docstring
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
                contexts_json=json.dumps(contexts) if contexts else "[]",
                global_commands_json=json.dumps(global_commands) if global_commands else "[]"
            )
            return result.description
        except Exception as e:
            logger.warning(f"Failed to generate workflow description: {e}")
            return "FastWorkflow automated workflow system."


# ============================================================================
# Post-Processor Main Class
# ============================================================================

class GenAIPostProcessor:
    """Main class for post-processing generated command files with AI-enhanced content."""
    
    def __init__(self):
        """Initialize the post-processor with DSPy configuration using FastWorkflow environment."""
        # Get model and API key from FastWorkflow environment
        self.model = fastworkflow.get_env_var("LLM_COMMAND_METADATA_GEN", default="mistral/mistral-small-latest")
        self.api_key = fastworkflow.get_env_var("LITELLM_API_KEY_COMMANDMETADATA_GEN", default=None)
        
        # Initialize DSPy with the configured model
        self._initialize_dspy()
        
        # Initialize DSPy modules
        self.field_generator = FieldMetadataGenerator()
        self.utterance_generator = UtteranceGenerator()
        self.docstring_generator = DocstringGenerator()
        self.workflow_generator = WorkflowDescriptionGenerator()
        
        # Cache for generated content to avoid redundant API calls
        self.cache = {}
    
    def _initialize_dspy(self):
        """Initialize DSPy with the configured model using LiteLLM."""
        try:
            # Use DSPy's LM class which supports LiteLLM model strings
            lm = dspy.LM(model=self.model, api_key=self.api_key, max_tokens=2000)
            dspy.settings.configure(lm=lm)
            logger.info(f"DSPy initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize DSPy: {e}")
            raise
    
    def process_workflow(self, workflow_path: str, classes: Dict[str, ClassInfo], 
                        functions: Dict[str, FunctionInfo] = None) -> bool:
        # sourcery skip: extract-method
        """Process all command files in the workflow with AI enhancements.
        
        Args:
            workflow_path: Path to the workflow directory
            classes: Dictionary of ClassInfo objects from the build phase
            functions: Dictionary of FunctionInfo objects (optional)
        
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        try:
            logger.info("Starting GenAI post-processing...")
            
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
                            'docstring': ""  # Will be filled later
                        }
                    
                    # Generate context handler docstring
                    self._generate_context_handler_docstring(context_path, context_dir, context_commands)
            
            # Process global commands (directly in _commands folder)
            for file_name in os.listdir(commands_dir):
                file_path = os.path.join(commands_dir, file_name)
                if os.path.isfile(file_path) and file_name.endswith('.py') and not file_name.startswith('_'):
                    command_name = file_name[:-3]  # Remove .py extension
                    # Find corresponding function info
                    func_info = functions.get(command_name) if functions else None
                    if self._process_command_file(file_path, command_name, None, func_info):
                        global_commands.append({
                            'name': command_name,
                            'docstring': func_info.docstring if func_info else ""
                        })
            
            # Generate workflow description
            self._generate_workflow_description(workflow_path, all_contexts, global_commands)
            
            logger.info("GenAI post-processing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during GenAI post-processing: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _process_context_commands(self, context_path: str, context_name: str, 
                                 class_info: Optional[ClassInfo]) -> List[Dict[str, str]]:
        """Process all command files in a context directory.
        
        Returns:
            List of command info dictionaries with name and docstring
        """
        commands = []
        
        for file_name in os.listdir(context_path):
            if file_name.endswith('.py') and not file_name.startswith('_'):
                file_path = os.path.join(context_path, file_name)
                command_name = file_name[:-3]  # Remove .py extension
                
                # Find corresponding method info if class_info is available
                method_info = None
                if class_info:
                    for method in class_info.methods:
                        if method.name == command_name:
                            method_info = method
                            break
                
                if self._process_command_file(file_path, command_name, context_name, method_info):
                    commands.append({
                        'name': command_name,
                        'docstring': method_info.docstring if method_info else ""
                    })
        
        return commands
    
    def _process_command_file(self, file_path: str, command_name: str, 
                             context_name: Optional[str], 
                             method_or_func_info: Optional[Any]) -> bool:
        """Process a single command file with AI enhancements.
        
        Args:
            file_path: Path to the command file
            command_name: Name of the command
            context_name: Name of the context (None for global commands)
            method_or_func_info: MethodInfo or FunctionInfo object
        
        Returns:
            bool: True if processing succeeded
        """
        try:
            return self._extracted_from__process_command_file_(
                file_path, method_or_func_info, command_name, context_name
            )
        except Exception as e:
            logger.error(f"Error processing command file {file_path}: {e}")
            logger.error(traceback.format_exc())
            return False

    # TODO Rename this here and in `_process_command_file`
    def _extracted_from__process_command_file_(self, file_path, method_or_func_info, command_name, context_name):
        logger.debug(f"Processing command file: {file_path}")

        # Read and parse the command file
        with open(file_path, 'r') as f:
            content = f.read()

        tree = ast.parse(content)

        signature_class = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name == "Signature"
            ),
            None,
        )
        if not signature_class:
            logger.warning(f"No Signature class found in {file_path}")
            return False

        # Extract input and output fields
        input_fields = self._extract_fields_from_class(signature_class, "Input")
        output_fields = self._extract_fields_from_class(signature_class, "Output")

        # Generate enhanced content
        enhanced_input_fields = []
        enhanced_output_fields = []

        # Process input fields
        for field_info in input_fields:
            metadata = self.field_generator(
                field_name=field_info['name'],
                field_type=field_info['type'],
                method_docstring=method_or_func_info.docstring if method_or_func_info else "",
                method_name=command_name,
                context_name=context_name or "global",
                is_input=True
            )

            enhanced_input_fields.append({
                'name': field_info['name'],
                'type': field_info['type'],
                'description': metadata.description,
                'examples': metadata.examples,
                'pattern': metadata.pattern
            })

        # Process output fields
        for field_info in output_fields:
            metadata = self.field_generator(
                field_name=field_info['name'],
                field_type=field_info['type'],
                method_docstring=method_or_func_info.docstring if method_or_func_info else "",
                method_name=command_name,
                context_name=context_name or "global",
                is_input=False
            )

            enhanced_output_fields.append({
                'name': field_info['name'],
                'type': field_info['type'],
                'description': metadata.description,
                'examples': metadata.examples,
                'pattern': metadata.pattern
            })

        # Generate utterances
        utterances = self.utterance_generator(
            command_name=command_name,
            command_docstring=method_or_func_info.docstring if method_or_func_info else "",
            input_fields=[{
                'name': f['name'],
                'type': f['type'],
                'description': f['description']
            } for f in enhanced_input_fields]
        )

        # Generate signature docstring
        signature_docstring = self.docstring_generator.generate_signature_docstring(
            command_name=command_name,
            input_fields=[{
                'name': f['name'],
                'type': f['type'],
                'description': f['description']
            } for f in enhanced_input_fields],
            output_fields=[{
                'name': f['name'],
                'type': f['type'],
                'description': f['description']
            } for f in enhanced_output_fields],
            context_name=context_name or "global",
            original_docstring=method_or_func_info.docstring if method_or_func_info else ""
        )

        # Update the AST with enhanced content
        updated_tree = self._update_command_file_ast(
            tree, enhanced_input_fields, enhanced_output_fields, 
            utterances, signature_docstring
        )

        # Write back the updated file
        updated_content = ast.unparse(updated_tree)
        with open(file_path, 'w') as f:
            f.write(updated_content)

        logger.debug(f"Successfully processed {file_path}")
        return True
    
    def _extract_fields_from_class(self, class_node: ast.ClassDef, 
                                  inner_class_name: str) -> List[Dict[str, str]]:
        """Extract field information from an inner class (Input or Output).
        
        Returns:
            List of field dictionaries with name and type
        """
        fields = []

        inner_class = next(
            (
                node
                for node in class_node.body
                if isinstance(node, ast.ClassDef) and node.name == inner_class_name
            ),
            None,
        )
        if not inner_class:
            return fields

        # Extract fields from the inner class
        for node in inner_class.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_name = node.target.id
                field_type = ast.unparse(node.annotation) if node.annotation else "Any"

                # Skip if it's a Field() assignment (we'll regenerate it)
                fields.append({
                    'name': field_name,
                    'type': field_type
                })

        return fields
    
    def _update_command_file_ast(self, tree: ast.Module, 
                                enhanced_input_fields: List[Dict],
                                enhanced_output_fields: List[Dict],
                                utterances: List[str],
                                signature_docstring: str) -> ast.Module:
        """Update the AST with enhanced content.
        
        Args:
            tree: The parsed AST of the command file
            enhanced_input_fields: List of enhanced input field dictionaries
            enhanced_output_fields: List of enhanced output field dictionaries
            utterances: List of generated utterances
            signature_docstring: Generated docstring for the Signature class
        
        Returns:
            Updated AST
        """
        # Find the Signature class
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "Signature":
                # Add or update the class docstring
                if signature_docstring:
                    # Ensure the first element is a docstring
                    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                        node.body[0].value.value = signature_docstring
                    else:
                        # Insert docstring at the beginning
                        docstring_node = ast.Expr(value=ast.Constant(value=signature_docstring))
                        node.body.insert(0, docstring_node)

                # Update Input and Output classes
                for class_node in node.body:
                    if isinstance(class_node, ast.ClassDef):
                        if class_node.name == "Input":
                            self._update_model_fields(class_node, enhanced_input_fields)
                        elif class_node.name == "Output":
                            self._update_model_fields(class_node, enhanced_output_fields)

                # Update utterances
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name) and target.id == "plain_utterances":
                                # Replace with new utterances
                                stmt.value = ast.List(
                                    elts=[ast.Constant(value=u) for u in utterances],
                                    ctx=ast.Load()
                                )
                                break

        return tree
    
    def _update_model_fields(self, class_node: ast.ClassDef, enhanced_fields: List[Dict]):
        """Update the fields in a BaseModel class with enhanced metadata.
        
        Args:
            class_node: The AST node for the Input or Output class
            enhanced_fields: List of enhanced field dictionaries
        """
        # Create a mapping of field names to enhanced data
        field_map = {f['name']: f for f in enhanced_fields}
        
        # Update existing field assignments
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_name = node.target.id
                if field_name in field_map:
                    field_data = field_map[field_name]
                    
                    # Create Field() call with enhanced metadata
                    field_call = ast.Call(
                        func=ast.Name(id='Field', ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(
                                arg='description',
                                value=ast.Constant(value=field_data['description'])
                            )
                        ]
                    )
                    
                    # Add examples if available
                    if field_data.get('examples'):
                        field_call.keywords.append(
                            ast.keyword(
                                arg='examples',
                                value=ast.List(
                                    elts=[ast.Constant(value=ex) for ex in field_data['examples']],
                                    ctx=ast.Load()
                                )
                            )
                        )
                    
                    # Add pattern for regex validation if provided
                    if field_data.get('pattern') and field_data['pattern']:
                        field_call.keywords.append(
                            ast.keyword(
                                arg='pattern',
                                value=ast.Constant(value=field_data['pattern'])
                            )
                        )
                    
                    # Update the assignment value
                    node.value = field_call
    
    def _generate_context_handler_docstring(self, context_path: str, 
                                           context_name: str, 
                                           commands: List[Dict[str, str]]):
        """Generate and write docstring for _<Context>.py file.
        
        Args:
            context_path: Path to the context directory
            context_name: Name of the context
            commands: List of command dictionaries
        """
        try:
            handler_file = os.path.join(context_path, f"_{context_name}.py")
            if not os.path.exists(handler_file):
                logger.debug(f"Context handler file not found: {handler_file}")
                return
            
            # Generate context docstring
            docstring = self.docstring_generator.generate_context_docstring(
                context_name=context_name,
                commands=commands
            )
            
            # Read the handler file
            with open(handler_file, 'r') as f:
                content = f.read()
            
            # Parse and update the AST
            tree = ast.parse(content)
            
            # Add or update module docstring
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
                # Update existing docstring
                tree.body[0].value.value = docstring
            else:
                # Insert new docstring at the beginning
                docstring_node = ast.Expr(value=ast.Constant(value=docstring))
                tree.body.insert(0, docstring_node)
            
            # Write back the updated file
            updated_content = ast.unparse(tree)
            with open(handler_file, 'w') as f:
                f.write(updated_content)
            
            logger.debug(f"Updated context handler docstring for {context_name}")
            
        except Exception as e:
            logger.error(f"Error generating context handler docstring: {e}")
    
    def _generate_workflow_description(self, workflow_path: str, 
                                      contexts: Dict[str, Dict],
                                      global_commands: List[Dict[str, str]]):
        """Generate and write workflow_description.txt file.
        
        Args:
            workflow_path: Path to the workflow directory
            contexts: Dictionary of context information
            global_commands: List of global command dictionaries
        """
        try:
            # Prepare context data for the generator
            context_list = list(contexts.values())
            
            # Generate workflow description
            description = self.workflow_generator(
                contexts=context_list,
                global_commands=global_commands
            )
            
            # Write to workflow_description.txt
            desc_file = os.path.join(workflow_path, "workflow_description.txt")
            with open(desc_file, 'w') as f:
                f.write(description)
            
            logger.info(f"Generated workflow description: {desc_file}")
            
        except Exception as e:
            logger.error(f"Error generating workflow description: {e}")


# ============================================================================
# Integration Functions
# ============================================================================

def run_genai_postprocessor(args, classes: Dict[str, ClassInfo], 
                          functions: Optional[Dict[str, FunctionInfo]] = None) -> bool:
    """Run the GenAI post-processor on generated command files.
    
    This function is called from the main build process after command files
    have been generated. Post-processing is always enabled and uses the
    LLM_COMMAND_METADATA_GEN and LITELLM_API_KEY_COMMANDMETADATA_GEN 
    environment variables for configuration.
    
    Args:
        args: Command-line arguments from the build tool
        classes: Dictionary of ClassInfo objects from analysis
        functions: Optional dictionary of FunctionInfo objects
    
    Returns:
        bool: True if post-processing succeeded, False otherwise
    """
    try:
        # Initialize the post-processor (uses environment variables internally)
        processor = GenAIPostProcessor()
        
        # Process the workflow
        workflow_path = args.workflow_folderpath
        return processor.process_workflow(workflow_path, classes, functions)
        
    except Exception as e:
        logger.error(f"GenAI post-processing failed: {e}")
        logger.error(traceback.format_exc())
        # Don't fail the entire build if post-processing fails
        return True
