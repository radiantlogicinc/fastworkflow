import dspy
import random
import re
import json
from typing import List, Optional, Union, Annotated, Dict, Any
from pydantic import Field
import Levenshtein  # Make sure to install this package
import litellm  # Import litellm instead of together
import fastworkflow

def normalize_text(text):
    """Normalize text by removing spaces, @ symbol, underscores, and converting to lowercase"""
    return "" if text is None else re.sub(r'[@\s_]', '', str(text).lower())

def normalized_levenshtein_distance(s1, s2):
    """Calculate normalized Levenshtein distance"""
    distance = Levenshtein.distance(s1, s2)
    max_length = max(len(s1), len(s2))
    return 0.0 if max_length == 0 else distance / max_length

def validate_parameters(utterance, params_dict, threshold=0.4):
    """
    Validate extracted parameters against the original utterance
    Returns a dictionary with validation results for each parameter
    """
    utterance = utterance.lower()
    results = {}

    for param_name, param_value in params_dict.items():
        # Skip None values and numeric values
        if param_value is None or isinstance(param_value, (int, float)):
            results[param_name] = {
                'value': param_value,
                'valid': True if param_value is not None else None,
                'confidence': 1.0 if param_value is not None else None
            }
            continue

        # Convert to string and normalize
        param_str = str(param_value).lower()

        # First check for exact substring match
        if param_str.lower() in utterance:
            results[param_name] = {
                'value': param_value,
                'valid': True,
                'confidence': 1.0,
                'match_type': 'exact'
            }
            continue

        # If no exact match, use fuzzy matching
        # Extract words and phrases from utterance to check against
        words = utterance.split()
        phrases = []
        for i in range(len(words)):
            phrases.extend(
                ' '.join(words[i:j])
                for j in range(i + 1, min(i + 6, len(words) + 1))
            )
        # Find best match among phrases
        best_match = None
        best_distance = float('inf')
        for phrase in phrases:
            norm_phrase = normalize_text(phrase)
            norm_param = normalize_text(param_str)
            if not norm_param: # Skip empty normalized params
                continue
            distance = normalized_levenshtein_distance(norm_phrase, norm_param)
            if distance < best_distance:
                best_distance = distance
                best_match = phrase

        confidence = 1.0 - best_distance
        results[param_name] = {
            'value': param_value,
            'valid': confidence >= (1.0 - threshold),
            'confidence': round(confidence, 2),
            'best_match': best_match
        }

    return results

def extract_field_details(field_annotations) -> List[Dict[str, Any]]:
    """
    Extract detailed information about fields from Pydantic model_fields.
    
    Works with both string annotations and actual model_fields dictionary.
    
    Args:
        field_annotations: Either a string of field annotations or the model_fields dict
        
    Returns:
        List of dictionaries with field details
    """
    field_details = []

    # Handle the case when field_annotations is already a dictionary (model_fields)
    if isinstance(field_annotations, dict):
        for field_name, field_info in field_annotations.items():
            # Extract field type
            field_type = str(field_info.annotation).replace("typing.", "")
            if "NoneType" in field_type:
                is_optional = True
                field_type = field_type.replace(" | NoneType", "").replace("NoneType | ", "")
            else:
                is_optional = field_info.is_required() is False

            # Extract description and examples from metadata
            description = ""
            examples = []
            pattern = None

            if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                schema_extra = field_info.json_schema_extra
                description = schema_extra.get('description', '')
                if 'examples' in schema_extra:
                    examples = schema_extra['examples']
                if 'pattern' in schema_extra:
                    pattern = schema_extra['pattern']

            # Add to field details
            field_details.append({
                "name": field_name,
                "type": field_type,
                "optional": is_optional,
                "description": description,
                "examples": examples,
                "pattern": pattern
            })

        return field_details

    # If field_annotations is a string, parse it (legacy support)
    field_names = re.findall(r'(\w+)(?=:)', field_annotations)

    for field_name in field_names:
        # Find this field's section in the annotations
        field_pattern = rf'{field_name}:\s*(.*?)(?=\w+:|$)'
        field_match = re.search(field_pattern, field_annotations, re.DOTALL)

        if not field_match:
            continue

        field_text = field_match.group(1).strip()

        # Check if optional
        is_optional = "Optional" in field_text or "None" in field_text

        # Extract field type - handle various formats
        if "Annotated" in field_text:
            # Extract the base type from Annotated
            type_match = re.search(r'Annotated\[\s*([^,\]]+)', field_text)
            field_type = type_match.group(1) if type_match else "str"
        elif "Optional" in field_text:
            # Extract the type from Optional
            type_match = re.search(r'Optional\[\s*([^,\]]+)', field_text)
            field_type = type_match.group(1) if type_match else "str"
        else:
            # Direct type
            field_type = field_text.split('=')[0].strip()

        # Clean up the type
        field_type = field_type.strip()

        # Extract description
        desc_match = re.search(r'description="([^"]+)"', field_text)
        description = desc_match.group(1) if desc_match else ""

        # Extract examples
        examples = []
        if examples_match := re.search(r'examples=\[(.*?)\]', field_text):
            examples_text = examples_match.group(1)
            if example_items := re.findall(r'"([^"]*)"', examples_text):
                examples = example_items
            else:
                # Try to safely evaluate the examples
                try:
                    examples = eval(f"[{examples_text}]")
                except:
                    # If evaluation fails, try to parse manually
                    examples = [item.strip() for item in examples_text.split(',')]

        # Extract pattern
        pattern_match = re.search(r'pattern=r?["\']([^"\']+)["\']', field_text)
        pattern = pattern_match.group(1) if pattern_match else None

        # Add to field details
        field_details.append({
            "name": field_name,
            "type": field_type,
            "optional": is_optional,
            "description": description,
            "examples": examples,
            "pattern": pattern
        })

    return field_details
    
# def transform_examples_to_dict_format(examples: List[str]) -> List[Dict]:
#     """
#     Transform examples from string format to dictionary format with fields and inputs.
    
#     Args:
#         examples: List of example strings in dspy.Example format
        
#     Returns:
#         List of dictionaries in the format {"fields": {...}, "inputs": [...]}
#     """
#     transformed_examples = []
    
#     for example in examples:
#         try:
#             # Extract all parameter assignments using regex
#             # This matches param_name=value patterns
#             fields = {}
            
#             # Find all field assignments
#             field_matches = re.findall(r'(\w+)=([^,\n\)]+)', example)
            
#             for field_name, field_value in field_matches:
#                 # Clean up the field value
#                 value = field_value.strip()
                
#                 # Store in fields dictionary
#                 fields[field_name] = value
            
#             # Extract inputs
#             inputs_match = re.search(r'with_inputs\(([^)]+)\)', example)
#             inputs = []
#             if inputs_match:
#                 inputs_text = inputs_match.group(1)
#                 # Extract quoted strings
#                 inputs = re.findall(r'"([^"]+)"', inputs_text)
            
#             # Add to transformed examples
#             transformed_examples.append({
#                 "fields": fields,
#                 "inputs": inputs
#             })
            
#         except Exception as e:
#             print(f"Error transforming example: {str(e)}")
    
#     return transformed_examples

def transform_examples_to_dict_format(examples: List[str]) -> List[Dict]:
    """
    Transform examples from string format to dictionary format with fields and inputs.
    Remove extra quotes from field values.
    """
    transformed_examples = []
    
    for example in examples:
        try:
            # Initialize fields dictionary
            fields = {}
            
            # Extract command with proper quote handling
            command_match = re.search(r'command="(.*?)"', example)
            if command_match:
                fields["command"] = command_match.group(1)
            
            # Find all other field assignments
            # We need to handle each type separately
            
            # Handle string values (quoted)
            string_matches = re.findall(r'(\w+)="([^"]*)"', example)
            for field_name, field_value in string_matches:
                if field_name != "command":  # Already handled
                    fields[field_name] = field_value
            
            # Handle boolean values
            bool_matches = re.findall(r'(\w+)=(True|False)', example)
            for field_name, field_value in bool_matches:
                fields[field_name] = field_value == "True"
            
            # Handle numeric values
            num_matches = re.findall(r'(\w+)=(\d+(?:\.\d+)?)', example)
            for field_name, field_value in num_matches:
                # Convert to int or float
                if '.' in field_value:
                    fields[field_name] = float(field_value)
                else:
                    fields[field_name] = int(field_value)
            
            # Handle None values
            none_matches = re.findall(r'(\w+)=None', example)
            for field_name in none_matches:
                fields[field_name] = None
            
            # Extract inputs
            inputs_match = re.search(r'with_inputs\(([^)]+)\)', example)
            inputs = []
            if inputs_match:
                inputs_text = inputs_match.group(1)
                # Extract quoted strings and remove quotes
                input_matches = re.findall(r'"([^"]+)"', inputs_text)
                inputs = input_matches
            
            # Add to transformed examples
            transformed_examples.append({
                "fields": fields,
                "inputs": inputs
            })
            
        except Exception as e:
            print(f"Error transforming example: {str(e)}")
    
    return transformed_examples

def generate_dspy_examples(
    field_annotations: str,
    command_name: str,
    num_examples: int = 10,
    validation_threshold: float = 0.4
) -> tuple[List[str], List[Dict]]:    # Updated return type to include rejected examples
    """
    Generate DSPy examples for parameter extraction based on field annotations.

    Args:
        field_annotations: String containing Pydantic field annotations
        command_name: Name of the command for which examples are generated
        num_examples: Number of examples to generate
        temperature: Temperature for generation
        model: Model to use for generation
        validation_threshold: Threshold for fuzzy matching validation

    Returns:
        Tuple of (valid examples list, rejected examples list)
    """

    model = fastworkflow.get_env_var("LLM_SYNDATA_GEN")
    api_key = fastworkflow.get_env_var("LITELLM_API_KEY_SYNDATA_GEN")
    temperature =  0.9
    # Extract detailed field information
    field_details = extract_field_details(field_annotations)

    # Create a section about each field with detailed information
    fields_section = ""
    if field_details:
        fields_section = "Fields to extract based on annotations:\n"
        for field in field_details:
            fields_section += f"""
        - {field['name']} ({field['type']})
          Description: {field['description']}
          {'Optional' if field['optional'] else 'Required'}
          Examples: {', '.join(repr(ex) for ex in field['examples']) if field['examples'] else 'None'}
          {f'Pattern: {field["pattern"]}' if field["pattern"] else ''}
        """

    # Construct the prompt with a focus on command name and optionality of parameters
    prompt = f"""
    You are a synthetic data generator for command processing.
    Generate {num_examples} realistic and diverse user utterances for the "{command_name}" command.

    For each utterance, create a complete DSPy Example with all parsed parameters.

    Here are field annotations that provide constraints and examples for fields:
    ```python
    {field_annotations}
    ```

    {fields_section}

    The output must strictly follow this structure:
    - Each example must be a "dspy.Example" object
    - Each example must have a "command" field with a user utterance as a string
    - Each example must include all relevant extracted parameters with appropriate values
    - String values must be in quotes
    - None values should be represented as Python None (not in quotes)
    - Boolean values should be True or False (not in quotes) IT SHOULD NOT BE NONE, IT IS EITHER TRUE OR FALSE
    - Generate parameter values do populate in some examples the values when default is none to keep variations.
    - Numeric values should be represented as numbers without quotes
    - Optional parameters should sometimes be None (absent) in the examples to create a variations.
    - Each example must end with .with_inputs("command")
    - KEEP VARIATIONS OF OPTIONAL PARAMETERS LIKE WITH VALUE NONE AND WITH SOME VALUE IN OTHER EXAMPLES WHEN ITS OPTIONAL
    VERY IMPORTANT: Make sure that all parameter values accurately reflect what's mentioned in the command utterance.
    Do not hallucinate parameter values that aren't clearly implied in the utterance.
    
    Examples should span a variety of scenarios with diverse parameter combinations.
    Ensure that some examples have all parameters, while others omit some optional parameters.
    Vary the way users might express their intent and include different phrasings.

    Generate {num_examples} diverse, syntactically correct examples for the "{command_name}" command:

    output should strictly be in this format:-

    dspy.Example(
        command="user utterance text",
        param1="value1",
        param2=None,  # If parameter is not mentioned in the utterance
        param3=True,  # For boolean parameters
        param4=42,    # For numeric parameters
    ).with_inputs("command")
    
    """

    # Call the model to generate examples using LiteLLM
    response = litellm.completion(
        model=model,
        api_key=api_key, 
        messages=[
            {"role": "system", "content": "You are a synthetic data generator. Generate realistic and diverse examples in the exact format requested."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000,
        temperature=temperature
    )

    # Extract the examples from the response
    examples_text = response.choices[0].message.content

    # Print the raw output for debugging
    print("Raw model output:")
    print(examples_text)
    print("\n" + "-"*50 + "\n")

    # Split the text into individual examples - improved extraction method
    examples = []

    # Find all lines that start with an example pattern
    example_lines = []
    current_example = []
    in_example = False

    # Look for lines containing example indicators
    for line in examples_text.split('\n'):
        # Check for the start of an example
        if "dspy.Example(" in line or line.strip().startswith("Example("):
            if in_example and current_example:
                example_lines.append('\n'.join(current_example))
                current_example = []
            in_example = True

        # If we're in an example, collect the line
        if in_example:
            current_example.append(line)

            # Check if this line completes the example
            if ".with_inputs" in line and ")" in line:
                example_lines.append('\n'.join(current_example))
                current_example = []
                in_example = False

    # Add the last example if there is one
    if in_example and current_example:
        example_lines.append('\n'.join(current_example))

    # Process each collected example
    for example_text in example_lines:
        # Clean up the example text
        example_text = example_text.strip()

        # Handle examples prefixed with numbers or comments
        lines = example_text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Remove numbered prefixes, e.g., "# Example 1:", "1.", etc.
            if line.strip().startswith('#') or line.strip()[0].isdigit():
                if "dspy.Example(" in line or "Example(" in line:
                    # Extract just the example part
                    example_part = line[line.find("Example("):]
                    cleaned_lines.append(example_part)
                else:
                    # Skip comment or number lines that don't contain examples
                    continue
            else:
                cleaned_lines.append(line)

        cleaned_example = '\n'.join(cleaned_lines)

        # Make sure the example uses dspy.Example
        if cleaned_example.startswith("Example("):
            cleaned_example = f"dspy.{cleaned_example}"

        # Ensure the example has no extra code or comments
        if "dspy.Example(" in cleaned_example and ".with_inputs" in cleaned_example:
            examples.append(cleaned_example)

    # Print the number of examples found
    print(f"Found {len(examples)} examples")

    # Validate examples using the fuzzy matching logic
    validated_examples = []
    rejected_examples = []

    for example in examples:
        try:
            # Extract command and parameters from the example
            # Use a regex to extract the command value
            command_match = re.search(r'command="(.*?)"', example)
            if not command_match:
                print(f"Couldn't extract command from example: {example[:100]}...")
                rejected_examples.append({
                    "example": example, 
                    "reason": "Command extraction failed",
                    "command": None,
                    "params": {}
                })
                continue

            command = command_match.group(1)

            # Extract parameters
            params = {}
            for field in field_details:
                field_name = field["name"]
                # Different regex patterns based on field type
                if field["type"] == "str":
                    # For string types, look for fieldname="value"
                    pattern = rf'{field_name}="(.*?)"'
                    if match := re.search(pattern, example):
                        params[field_name] = match.group(1)
                    else:
                        # Check if it's explicitly None
                        none_pattern = rf'{field_name}=None'
                        if re.search(none_pattern, example):
                            params[field_name] = None
                elif field["type"] == "int":
                    # For int types, look for fieldname=123
                    pattern = rf'{field_name}=(\d+)'
                    if match := re.search(pattern, example):
                        params[field_name] = int(match.group(1))
                    else:
                        # Check if it's explicitly None
                        none_pattern = rf'{field_name}=None'
                        if re.search(none_pattern, example):
                            params[field_name] = None
                elif field["type"] == "float":
                    # For float types, look for fieldname=123.45
                    pattern = rf'{field_name}=(\d+\.\d+)'
                    if match := re.search(pattern, example):
                        params[field_name] = float(match.group(1))
                    else:
                        # Check if it's explicitly None
                        none_pattern = rf'{field_name}=None'
                        if re.search(none_pattern, example):
                            params[field_name] = None
                elif field["type"] == "bool" or field["type"].endswith("bool"):
                    # For boolean types, look for fieldname=True or fieldname=False
                    pattern = rf'{field_name}=(True|False)'
                    if match := re.search(pattern, example):
                        params[field_name] = match.group(1) == "True"
                    else:
                        # Check if it's explicitly None
                        none_pattern = rf'{field_name}=None'
                        if re.search(none_pattern, example):
                            params[field_name] = None

            # Validate the extracted parameters against the command
            validation_results = validate_parameters(command, params, threshold=validation_threshold)

            # Check if all required parameters are valid
            invalid_params = []
            for field in field_details:
                if not field["optional"] and field["name"] in validation_results:
                    result = validation_results[field["name"]]
                    if not result.get("valid", False) and result["value"] is not None:
                        invalid_params.append({
                            "param": field["name"],
                            "value": result["value"],
                            "confidence": result.get("confidence"),
                            "best_match": result.get("best_match")
                        })

            # Also check if any optional parameters that have values are invalid
            for field in field_details:
                if field["optional"] and field["name"] in validation_results:
                    result = validation_results[field["name"]]
                    if result["value"] is not None and not result.get("valid", False):
                        invalid_params.append({
                            "param": field["name"],
                            "value": result["value"],
                            "confidence": result.get("confidence"),
                            "best_match": result.get("best_match")
                        })

            if invalid_params:
                rejection_reason = {
                    "example": example,
                    "command": command,
                    "params": params,
                    "invalid_params": invalid_params,
                    "validation_results": validation_results
                }
                rejected_examples.append(rejection_reason)
                print(f"Rejected example: '{command}' - Invalid parameters: {invalid_params}")
            else:
                validated_examples.append(example)

        except Exception as e:
            print(f"Error processing example: {str(e)}")
            rejected_examples.append({
                "example": example, 
                "reason": str(e),
                "command": command if 'command' in locals() else None,
                "params": params if 'params' in locals() else {}
            })

    print(f"Validated {len(validated_examples)} examples, rejected {len(rejected_examples)} examples")

    # If we have rejected examples, save them for analysis
    if rejected_examples:
        with open("rejected_examples.json", "w") as f:
            json.dump(rejected_examples, f, indent=2)
    dict_examples = transform_examples_to_dict_format(examples)

    return dict_examples, rejected_examples
            
    # return validated_examples, rejected_examples

def save_examples_to_file(examples: List[str], filename: str = "dspy_examples.py"):
    """Save generated examples to a Python file"""
    with open(filename, "w") as f:
        f.write("import dspy\n\n")
        f.write("examples = [\n")
        for example in examples:
            f.write(f"    {example},\n")
        f.write("]\n")

def save_examples_to_json(examples: List[str], command_name: str, filename: str = "dspy_examples.json"):
    """Save generated examples to a JSON file"""
    import json

    # Format examples properly as strings
    formatted_examples = list(examples)

    data = {
        "command_name": command_name,
        "examples": formatted_examples
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    
