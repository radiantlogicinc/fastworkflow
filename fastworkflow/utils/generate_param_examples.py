import ast
import dspy
import hashlib
import random
import re
import json
from enum import Enum
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
)
from pydantic import Field
import Levenshtein  # Make sure to install this package
import litellm  # Import litellm instead of together
import fastworkflow
from fastworkflow.train.determinism import get_training_seed
from fastworkflow.train.param_example_cache import (
    compute_fingerprint,
    get_param_example_cache,
    run_stable_form,
)
from fastworkflow.train.utterance_cache import (
    PRODUCTION_COMPLETION_BACKEND,
    Fingerprint,
    callable_identity,
    source_digest,
)
from fastworkflow.utils.logging import logger

# The sampling temperature for parameter-example generation, lifted out of the call
# site so that the cache fingerprint can name it explicitly (bd fix-czb).
#
# IT IS DELIBERATELY LEFT AT 0.9 AND SHOULD NOT BE LOWERED WITHOUT A MEASUREMENT.
# Lowering it would be a change to training-data QUALITY masquerading as a
# determinism fix. Determinism here is delivered by the cache, not by the
# temperature: reuse makes repeat runs identical, whereas temperature 0 would only
# narrow the distribution of the FIRST draw, and no hosted provider guarantees
# bitwise-identical completions at temperature 0 anyway (batching, kernel scheduling
# and silent model revisions all move the output). Meanwhile the prompt explicitly
# asks for "diverse", "varied" examples that "span a variety of scenarios" — high
# temperature is doing work here, and nothing has measured what collapsing it costs
# parameter-extraction accuracy.
DSPY_EXAMPLE_TEMPERATURE: float = 0.9

def normalize_text(text):
    """Normalize text by removing spaces, @ symbol, underscores, and converting to lowercase"""
    return "" if text is None else re.sub(r'[@\s_]', '', str(text).lower())

def normalized_levenshtein_distance(s1, s2):
    """Calculate normalized Levenshtein distance"""
    distance = Levenshtein.distance(s1, s2)
    max_length = max(len(s1), len(s2))
    return 0.0 if max_length == 0 else distance / max_length


def _candidate_phrases(utterance: str, param_value: str) -> list[str]:
    """Return short and value-length windows for fuzzy parameter matching.

    The original matcher considered at most five words. That works for identifiers and
    names, but it structurally rejects valid long free-text values as soon as punctuation
    or a light paraphrase defeats the exact-substring check: no five-word window can be
    close to a ten-word message. Keep the original short windows, and add windows within
    two words of the extracted value's own length. This fixes the measured long-message
    false rejections without relaxing the global distance threshold for short identifiers.
    """
    words = utterance.split()
    if not words:
        return []

    short_window_lengths = range(1, min(5, len(words)) + 1)
    param_word_count = max(1, len(param_value.split()))
    value_window_lengths = range(
        max(1, param_word_count - 2),
        min(len(words), param_word_count + 2) + 1,
    )
    window_lengths = sorted(set(short_window_lengths) | set(value_window_lengths))
    return [
        " ".join(words[start:start + window_length])
        for window_length in window_lengths
        for start in range(len(words) - window_length + 1)
    ]


def validate_parameters(utterance, params_dict, threshold=0.4):
    """
    Validate extracted parameters against the original utterance
    Returns a dictionary with validation results for each parameter
    """
    utterance = utterance.lower()
    results = {}

    for param_name, param_value in params_dict.items():
        values_to_validate = _parameter_values(param_value)
        if not values_to_validate:
            results[param_name] = {
                'value': param_value,
                'valid': True,
                'confidence': 1.0
            }
            continue

        value_results = [
            _validate_parameter_value(utterance, value, threshold)
            for value in values_to_validate
        ]
        least_confident = min(
            value_results,
            key=lambda result: (
                result.get("confidence")
                if result.get("confidence") is not None
                else 1.0
            ),
        )
        results[param_name] = {
            'value': param_value,
            'valid': all(result.get("valid") is not False for result in value_results),
            'confidence': least_confident.get("confidence"),
        }
        if "match_type" in least_confident:
            results[param_name]["match_type"] = least_confident["match_type"]
        if "best_match" in least_confident:
            results[param_name]["best_match"] = least_confident["best_match"]

    return results


def _parameter_values(param_value: Any) -> list[Any]:
    """Flatten supported containers into values that should occur in the utterance."""
    if isinstance(param_value, dict):
        return [
            nested_value
            for value in param_value.values()
            for nested_value in _parameter_values(value)
        ]
    if isinstance(param_value, list):
        return [
            nested_value
            for value in param_value
            for nested_value in _parameter_values(value)
        ]
    return [param_value]


def _validate_parameter_value(
    utterance: str, param_value: Any, threshold: float
) -> Dict[str, Any]:
    """Validate one scalar parameter value against the original utterance."""
    # Skip None values and numeric values
    if param_value is None or isinstance(param_value, (int, float)):
        return {
            'value': param_value,
            'valid': True if param_value is not None else None,
            'confidence': 1.0 if param_value is not None else None
        }

    # Convert to string and normalize
    param_str = str(param_value).lower()

    # First check for exact substring match
    if param_str.lower() in utterance:
        return {
            'value': param_value,
            'valid': True,
            'confidence': 1.0,
            'match_type': 'exact'
        }

    # If no exact match, use fuzzy matching
    # Extract short phrases and windows close to the value's own length.
    phrases = _candidate_phrases(utterance, param_str)
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

    confidence = 0.0 if best_distance == float('inf') else 1.0 - best_distance
    return {
        'value': param_value,
        'valid': confidence >= (1.0 - threshold),
        'confidence': round(confidence, 2),
        'best_match': best_match
    }


def extract_field_details(
    field_annotations: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extract detailed information about fields from Pydantic model_fields.

    Args:
        field_annotations: a parameter model's ``model_fields`` mapping

    Returns:
        List of dictionaries with field details

    A regex parser for a STRINGIFIED ``model_fields`` repr used to live here as
    "legacy support", and it was a fail-open path rather than a fallback (bd
    fix-k0i.46). Downstream, `generate_dspy_examples` derives
    ``model_fields = field_annotations if isinstance(field_annotations, dict) else {}``,
    so under a string every type check and every enum-membership check found an empty
    mapping and silently passed — the exact validation hole bd fix-b8h closed for
    dicts stayed wide open through this function's own declared parameter type. It
    also guessed ``required`` from whether the substring "Optional" appeared. No
    caller has ever passed a string: the only production call site
    (`train/__main__.py`) passes ``fields.model_fields``. So the parser is gone and a
    non-mapping now raises, because failing at build time beats shipping a command
    whose parameter examples were never validated.
    """
    if not isinstance(field_annotations, dict):
        raise TypeError(
            f"extract_field_details needs a parameter model's model_fields mapping, "
            f"not {type(field_annotations).__name__}. Pass "
            f"YourSignature.Input.model_fields; a stringified repr cannot be validated "
            f"against, so accepting one would accept every generated example unchecked."
        )

    field_details = []

    for field_name, field_info in field_annotations.items():
        # Extract field type
        annotation = field_info.annotation
        annotation_args = get_args(annotation)
        is_union = get_origin(annotation) in (Union, UnionType)
        non_none_types = [
            arg for arg in annotation_args if arg is not type(None)
        ]
        is_optional = (
            (is_union and len(non_none_types) != len(annotation_args))
            or field_info.is_required() is False
        )
        is_required = field_info.is_required()
        base_annotation = (
            non_none_types[0]
            if is_union and len(non_none_types) == 1
            else annotation
        )
        field_type = (
            str(base_annotation).replace("typing.", "")
            if get_origin(base_annotation) is not None
            else getattr(base_annotation, "__name__", str(base_annotation))
        )

        # Extract description and examples from metadata
        description = ""
        examples = []
        pattern = None
        enum_values = []

        if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
            schema_extra = field_info.json_schema_extra
            description = schema_extra.get('description', '')
            if 'examples' in schema_extra:
                examples = schema_extra['examples']
            if 'pattern' in schema_extra:
                pattern = schema_extra['pattern']
            enum_values = list(schema_extra.get('enum', []))
        description = field_info.description or description
        examples = list(field_info.examples or examples)
        for metadata in field_info.metadata:
            if pattern is None and hasattr(metadata, "pattern"):
                pattern = metadata.pattern

        # Add to field details
        field_details.append({
            "name": field_name,
            "type": field_type,
            "optional": is_optional,
            "required": is_required,
            "description": description,
            "examples": examples,
            "pattern": pattern,
            "enum": enum_values,
        })

    return field_details


def _is_supported_literal(value: Any) -> bool:
    """Return whether a literal can round-trip through the runtime JSON schema."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        return all(_is_supported_literal(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_supported_literal(item)
            for key, item in value.items()
        )
    return False


def _parse_dspy_example(example: str) -> Dict[str, Any]:
    """Parse one generated DSPy example without executing model-provided text."""
    try:
        expression = ast.parse(example.strip(), mode="eval").body
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid example syntax: {exc}") from exc

    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Attribute)
        or expression.func.attr != "with_inputs"
        or expression.keywords
    ):
        raise ValueError("example must end with .with_inputs(...)")

    example_call = expression.func.value
    if not isinstance(example_call, ast.Call) or example_call.args:
        raise ValueError("Example fields must use keyword arguments")
    is_example_constructor = (
        isinstance(example_call.func, ast.Name)
        and example_call.func.id == "Example"
    ) or (
        isinstance(example_call.func, ast.Attribute)
        and example_call.func.attr == "Example"
        and isinstance(example_call.func.value, ast.Name)
        and example_call.func.value.id == "dspy"
    )
    if not is_example_constructor:
        raise ValueError("example must call dspy.Example(...)")

    fields: Dict[str, Any] = {}
    for keyword in example_call.keywords:
        if keyword.arg is None:
            raise ValueError("expanded keyword arguments are not supported")
        if keyword.arg in fields:
            raise ValueError(f"duplicate field {keyword.arg!r}")
        try:
            value = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"field {keyword.arg!r} must be a Python literal"
            ) from exc
        if not _is_supported_literal(value):
            raise ValueError(
                f"field {keyword.arg!r} must use JSON-compatible literals"
            )
        fields[keyword.arg] = value

    inputs = []
    for input_node in expression.args:
        try:
            input_name = ast.literal_eval(input_node)
        except (ValueError, TypeError) as exc:
            raise ValueError("input names must be string literals") from exc
        if not isinstance(input_name, str):
            raise ValueError("input names must be string literals")
        inputs.append(input_name)
    if inputs != ["command"]:
        raise ValueError("with_inputs arguments must be exactly ['command']")

    return {"fields": fields, "inputs": inputs}


def _value_matches_annotation(value: Any, annotation: Any) -> bool:
    """Return whether a parsed literal has the shape required by an annotation."""
    if annotation in (Any, object):
        return True
    if annotation is None or annotation is type(None):
        return value is None

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _value_matches_annotation(value, args[0])
    if origin in (Union, UnionType):
        return any(_value_matches_annotation(value, arg) for arg in args)
    if origin is Literal:
        return any(
            type(value) is type(literal_value) and value == literal_value
            for literal_value in args
        )
    if origin is list:
        item_annotation = args[0] if args else Any
        return isinstance(value, list) and all(
            _value_matches_annotation(item, item_annotation) for item in value
        )
    if origin is dict:
        key_annotation, value_annotation = args if len(args) == 2 else (Any, Any)
        return isinstance(value, dict) and all(
            _value_matches_annotation(key, key_annotation)
            and _value_matches_annotation(item, value_annotation)
            for key, item in value.items()
        )

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return any(
            value in (member.value, member.name)
            for member in annotation
        )
    if annotation is bool:
        return type(value) is bool
    if annotation is int:
        return type(value) is int
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation is str:
        return isinstance(value, str)
    return isinstance(annotation, type) and isinstance(value, annotation)


def _field_value_error(
    field_name: str,
    value: Any,
    annotation: Any,
    enum_values: list[Any],
) -> Optional[str]:
    """Describe a malformed field value, or return None when it is supported."""
    if not _value_matches_annotation(value, annotation):
        return (
            f"{field_name!r} has value {value!r}, which does not match "
            f"{annotation!r}"
        )
    if enum_values and value not in enum_values:
        return (
            f"{field_name!r} has value {value!r}; expected one of "
            f"{enum_values!r}"
        )
    return None


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
            parsed_example = _parse_dspy_example(example)
            fields = parsed_example["fields"]
            
            # Extract command with proper quote handling
            # Find all other field assignments

            # We need to handle each type separately
            
            # Handle string values (quoted)
            # Handle boolean values
            # Handle numeric values
            # Convert to int or float
            
            # Handle None values
            # Extract inputs
            inputs = parsed_example["inputs"]
            # Extract quoted strings and remove quotes
            
            # Add to transformed examples
            transformed_examples.append({
                "fields": fields,
                "inputs": inputs
            })
            
        except Exception as e:
            print(f"Error transforming example: {str(e)}")
    
    return transformed_examples

def _address_free(text: str) -> str:
    """Strip the memory address `object.__repr__` embeds, wherever it appears in *text*.

    The prompt interpolates ``str(field_annotations)`` verbatim inside a ```python
    fence, and pydantic's ``FieldInfo`` repr nests the repr of anything in
    ``json_schema_extra`` — so a field declared
    ``Field(json_schema_extra={"examples": [Foo()]})`` puts ``<Foo object at 0x7f...>``
    into the prompt through that block as well as through the "Examples:" line that bd
    fix-r4p names. Fixing only the named line would leave the address in the prompt, so
    the reproducibility it costs would be unchanged.

    A textual strip rather than a structural re-render, deliberately: rendering
    ``model_fields`` ourselves would change the prompt for every workflow in existence,
    which is a training-data QUALITY change wearing a determinism fix's clothes. This
    matches nothing at all in a workflow whose examples are JSON-native — which is every
    shipped one — so their prompts stay byte-identical.

    The pattern is a literal inside this function on purpose. A module-level compiled
    constant would sit outside `prompt_source_digest`, and editing it would then change
    the prompt without invalidating a single cache entry: the exact shape of hole this
    function was added to close.
    """
    return re.sub(r" at 0x[0-9a-fA-F]+", "", text)


def render_field_examples(
    field_name: str, examples: List[Any], command_name: str
) -> str:
    """Render one field's declared examples for the prompt's "Examples:" line.

    Was ``', '.join(repr(ex) for ex in examples)``. `repr` of an object inheriting
    ``object.__repr__`` embeds ``id(self)``, so the model was shown a memory address and
    conditioned on it, and it differed on every run (bd fix-r4p).

    Canonicalised through `run_stable_form` — the same substitutions the cache key's
    digest makes — and then ``repr``d, which keeps the rendering Python-shaped. That
    matters because the prompt demands Python output three lines further down ("None
    values should be represented as Python None", "Boolean values should be True or
    False"): rendering ``null`` and ``true`` here would contradict it. For every
    JSON-native example this produces exactly the text it produced before.

    A value with no run-stable representation is rendered as its type, and reported at
    WARNING. Both halves are needed: the type is stable and address-free, and the
    warning is the only place a developer can learn that the model was shown a
    placeholder instead of an example. It is a coercion, but not a silent one — and it
    is not a raise, because `extract_field_details` accepts such a value today, a
    command carrying one still trains (only uncached), and a prompt renderer must not be
    the thing that fails a multi-hour run.
    """
    if not examples:
        return "None"
    rendered: List[str] = []
    for example in examples:
        try:
            rendered.append(repr(run_stable_form(example)))
        except TypeError:
            type_name = f"{type(example).__module__}.{type(example).__qualname__}"
            logger.warning(
                f"Field '{field_name}' of command '{command_name}' declares an example "
                f"of type {type_name}, which has no run-stable representation. The "
                f"prompt shows the model the type name instead of a value; two runs "
                f"would otherwise have been sent different memory addresses. Make the "
                f"example JSON-native (a string, number, boolean, None, list or dict) "
                f"to have it reach the model."
            )
            rendered.append(f"<{type_name}>")
    return ", ".join(rendered)


# Functions whose source text is hashed into the parameter-example cache key. Two
# kinds are in here and both matter:
#
#   * PROMPT-BEARING  -- generate_dspy_examples holds the prompt template itself,
#     extract_field_details produces the "Fields to extract" block rendered into it, and
#     render_field_examples / _address_free decide the text of two parts of that block.
#     Edit any of them and the LLM is asked a different question.
#   * PAYLOAD-SHAPING -- transform_examples_to_dict_format produces the accepted
#     examples that get cached, and validate_parameters (with the two text helpers it
#     calls) decides which examples are rejected. Edit any of them and the same LLM
#     response would be written to disk differently.
#
# Deliberately coarse, exactly as R6's is: ANY edit to one of these, including to a
# comment, invalidates every entry. Over-invalidation costs money; under-invalidation
# costs trust in the measurement. `_UNDIGESTED_FUNCTIONS` names the rest of the module
# explicitly so that adding a function here is a decision rather than an oversight,
# and `test_param_example_cache.py` fails if a new one is in neither list.
def _digested_functions() -> tuple:
    """The functions above, resolved late because one of them is defined below."""
    return (
        generate_dspy_examples,
        extract_field_details,
        render_field_examples,
        _address_free,
        transform_examples_to_dict_format,
        _parse_dspy_example,
        _is_supported_literal,
        _value_matches_annotation,
        _field_value_error,
        _parameter_values,
        _validate_parameter_value,
        validate_parameters,
        normalized_levenshtein_distance,
        normalize_text,
        _candidate_phrases,
        canonicalized,
    )


def canonicalized(value):
    """Recursively sort every mapping's keys, leaving lists and scalars alone.

    Applied to both halves of what `generate_dspy_examples` returns, on BOTH the
    generation path and the cache-hit path, so that the two produce byte-identical
    `<command>_param_labeled.json`.

    It is here because of a real failure, not a hypothetical one. A generated example
    is assembled by `transform_examples_to_dict_format` in the order the regexes
    happen to match — `command`, then strings, then booleans, then numbers — while a
    cached one comes back through `json.dump(..., sort_keys=True)`, which sorts nested
    keys too. The two dicts compare EQUAL and serialise to DIFFERENT bytes. On
    `examples/hello_world` that difference is invisible, because `command` <
    `first_number` < `second_number` is already alphabetical; on
    `examples/messaging_app_4` it left 3 of 5 artifacts differing between two runs
    with every example identical. Canonicalising both paths removes the whole class of
    problem rather than making the cache's serialiser match one particular writer.
    """
    if isinstance(value, dict):
        return {key: canonicalized(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [canonicalized(item) for item in value]
    return value

# Functions in this module that provably cannot change what is generated or what is
# written: two file-writing helpers that training never calls, and the cache plumbing
# itself (whose behaviour is already covered by the fingerprint it computes).
_UNDIGESTED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "save_examples_to_file",
        "save_examples_to_json",
        "param_example_fingerprint",
        "prompt_source_digest",
        "_digested_functions",
    }
)


def prompt_source_digest() -> str:
    """Digest of every function that decides what is asked for and what is stored."""
    combined = "|".join(source_digest(func) for func in _digested_functions())
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def param_example_fingerprint(
    field_annotations,
    command_name: str,
    num_examples: int,
    validation_threshold: float,
    model: Optional[str],
    field_details: Optional[List[Dict[str, Any]]] = None,
    completion_fn: Optional[Callable] = None,
) -> Optional[Fingerprint]:
    """Fingerprint this command's parameter-example configuration for the cache.

    Everything the LLM sees is in here, plus the two inputs that decide how its
    response is written to disk (`validation_threshold` and the payload-shaping
    source). See `param_example_cache.compute_fingerprint` for the per-field
    rationale.

    Returns None — meaning "this command cannot be cached this run" — when a field
    detail has no run-stable representation to digest. That is deliberately loud and
    deliberately not fatal: a cache must never be the reason a training run fails, but
    a command that silently re-draws its examples every run while looking like a first
    run is precisely the invisible cost bd fix-k0i.46 reports.

    The proxy base is read with a code default so an absent one does not log a warning
    per command; `LITELLM_PROXY_API_BASE` is an env-file setting in this project, never
    a shell export.
    """
    api_base = fastworkflow.get_env_var("LITELLM_PROXY_API_BASE", str, default="")
    if field_details is None:
        field_details = extract_field_details(field_annotations)
    try:
        return compute_fingerprint(
            command_name=str(command_name),
            field_annotations_text=str(field_annotations),
            field_details=field_details,
            num_examples=num_examples,
            validation_threshold=validation_threshold,
            temperature=DSPY_EXAMPLE_TEMPERATURE,
            model=model,
            api_base=api_base or None,
            completion_backend=callable_identity(
                completion_fn, PRODUCTION_COMPLETION_BACKEND),
            generator_source_digest=prompt_source_digest(),
        )
    except TypeError as exc:
        logger.warning(
            f"Cannot compute a parameter-example cache key for command "
            f"'{command_name}': {exc} Its examples will be regenerated from the LLM on "
            f"every training run, and two runs at the same seed will write different "
            f"{command_name}_param_labeled.json files."
        )
        return None


def generate_dspy_examples(
    field_annotations: Dict[str, Any],
    command_name: str,
    num_examples: int = 10,
    validation_threshold: float = 0.4,
    seed: Optional[int] = None,
    completion_fn: Optional[Callable] = None,
) -> tuple[List[str], List[Dict]]:    # Updated return type to include rejected examples
    """
    Generate DSPy examples for parameter extraction based on field annotations.

    Args:
        field_annotations: a parameter model's ``model_fields`` mapping. Declared as
            ``str`` until bd fix-k0i.46: a string made every generated value pass
            validation unchecked, so it now raises in `extract_field_details`.
        command_name: Name of the command for which examples are generated
        num_examples: Number of examples to generate
        temperature: Temperature for generation
        model: Model to use for generation
        validation_threshold: Threshold for fuzzy matching validation

    Returns:
        Tuple of (valid examples list, rejected examples list)

    When the trainer has installed a `ParamExampleCache` (bd fix-czb) and an entry
    matches this command's fingerprint at this seed, the LLM is not called at all.
    Reuse is what makes two runs at the same seed write the same
    `<command>_param_labeled.json`; nothing in this request is seeded, and the
    sampling temperature is deliberately high, so no seed could make it reproducible.

    `seed` defaults to the configured `TRAINING_SEED`. `completion_fn` exists so the
    whole path can be driven without a network call; production leaves it None, and it
    is named in the fingerprint so an injected generator can never collide with the
    real one.
    """

    model = fastworkflow.get_env_var("LLM_SYNDATA_GEN")
    api_key = fastworkflow.get_env_var("LITELLM_API_KEY_SYNDATA_GEN")
    temperature =  DSPY_EXAMPLE_TEMPERATURE
    if seed is None:
        seed = get_training_seed()
    if completion_fn is None:
        completion_fn = litellm.completion
    # Extract detailed field information
    field_details = extract_field_details(field_annotations)

    # Consulted before the prompt is even built: a hit spends no tokens and, more to
    # the point, returns the SAME examples as the run that produced the entry.
    cache = get_param_example_cache()
    fingerprint = (
        param_example_fingerprint(
            field_annotations,
            command_name,
            num_examples,
            validation_threshold,
            model,
            field_details=field_details,
            completion_fn=completion_fn if completion_fn is not litellm.completion else None,
        )
        if cache is not None and cache.enabled
        else None
    )
    if cache is not None and fingerprint is not None:
        if entry := cache.lookup(fingerprint, seed):
            logger.info(
                f"Reusing {len(entry.valid_examples)} cached DSPy parameter examples "
                f"for command '{command_name}' (seed {seed}, variant "
                f"{fingerprint.variant_key})"
            )
            # `rejected_examples.json` is deliberately NOT rewritten here. It is a
            # debugging dump of a generation, written to the current working
            # directory and overwritten by every command; on a hit there was no
            # generation to dump.
            return (
                canonicalized(list(entry.valid_examples)),
                canonicalized(list(entry.rejected_examples)),
            )

    # Create a section about each field with detailed information
    fields_section = ""
    if field_details:
        fields_section = "Fields to extract based on annotations:\n"
        for field in field_details:
            fields_section += f"""
        - {field['name']} ({field['type']})
          Description: {field['description']}
          {'Required' if field['required'] else 'Optional'}
          Examples: {render_field_examples(field['name'], field['examples'], command_name)}
          {f'Allowed values: {field["enum"]}' if field["enum"] else ''}
          {f'Pattern: {field["pattern"]}' if field["pattern"] else ''}
        """

    # Construct the prompt with a focus on command name and optionality of parameters
    prompt = f"""
    You are a synthetic data generator for command processing.
    Generate {num_examples} realistic and diverse user utterances for the "{command_name}" command.

    For each utterance, create a complete DSPy Example with all parsed parameters.

    Here are field annotations that provide constraints and examples for fields:
    ```python
    {_address_free(str(field_annotations))}
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
    response = completion_fn(
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
            stripped_line = line.strip()
            if stripped_line and (
                stripped_line.startswith('#') or stripped_line[0].isdigit()
            ):
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
        command = None
        params = {}
        try:
            # Extract command and parameters from the example
            # Use a regex to extract the command value
            parsed_example = _parse_dspy_example(example)
            command = parsed_example["fields"].get("command")
            if not isinstance(command, str):
                print(f"Couldn't extract command from example: {example[:100]}...")
                rejected_examples.append({
                    "example": example, 
                    "reason": "Command extraction failed",
                    "command": None,
                    "params": {}
                })
                continue

            # Extract parameters
            declared_fields = {
                "command", *(field["name"] for field in field_details)
            }
            if unknown_fields := sorted(
                set(parsed_example["fields"]) - declared_fields
            ):
                unknown_field_names = ", ".join(repr(name) for name in unknown_fields)
                rejected_examples.append({
                    "example": example,
                    "reason": f"Unknown fields: {unknown_field_names}",
                    "command": command,
                    "params": {},
                    "unknown_fields": unknown_fields,
                })
                continue

            params = {
                field["name"]: parsed_example["fields"][field["name"]]
                for field in field_details
                if field["name"] in parsed_example["fields"]
            }
            # `extract_field_details` has already rejected a non-mapping, so this is the
            # real model_fields. It used to fall back to `{}` for a string, which made
            # every malformed-value check below a no-op instead of a check (bd fix-k0i.46).
            model_fields = field_annotations
            if missing_required := [
                field["name"]
                for field in field_details
                if field["required"] and field["name"] not in params
            ]:
                rejected_examples.append({
                    "example": example,
                    "reason": "Missing required parameters",
                    "command": command,
                    "params": params,
                    "missing_required": missing_required,
                })
                continue

            malformed_params = []
            for field in field_details:
                field_name = field["name"]
                # Different regex patterns based on field type
                # For string types, look for fieldname="value"
                # Check if it's explicitly None
                # For int types, look for fieldname=123
                # Check if it's explicitly None
                # For float types, look for fieldname=123.45
                # Check if it's explicitly None
                # For boolean types, look for fieldname=True or fieldname=False
                # Check if it's explicitly None
                # The structured parser handles all of these forms before type checks.
                if field_name not in params or field_name not in model_fields:
                    continue
                annotation = model_fields[field_name].annotation
                if error := _field_value_error(
                    field_name, params[field_name], annotation, field["enum"]
                ):
                    malformed_params.append({
                        "param": field_name,
                        "value": params[field_name],
                        "error": error,
                    })

            if malformed_params:
                rejected_examples.append({
                    "example": example,
                    "reason": "Malformed parameter values",
                    "command": command,
                    "params": params,
                    "invalid_params": malformed_params,
                })
                continue

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

    # Rejected examples are returned to the trainer, written beside the accepted examples,
    # and stored in the parameter-example cache. Do not also drop a process-global
    # `rejected_examples.json` into the current working directory: every command overwrote
    # the previous command's file, and normal training left debug debris at the repo root.
    if rejected_examples:
        logger.info(
            f"Rejected {len(rejected_examples)} DSPy parameter example(s) for "
            f"command '{command_name}'; details are retained in the command artifact."
        )
    # Keep the runtime's {"fields": ..., "inputs": ...} schema while excluding the
    # source strings that validation rejected.
    dict_examples = transform_examples_to_dict_format(validated_examples)

    # Canonicalised BEFORE caching and before returning, so a run that generated and a
    # run that reused write the same bytes. See `canonicalized`.
    dict_examples = canonicalized(dict_examples)
    rejected_examples = canonicalized(rejected_examples)

    # A command that accepted NOTHING ships an empty few-shot corpus: the runtime falls
    # back to zero-shot parameter extraction for it, silently and for the whole life of
    # the artifact. "Validated 0 examples, rejected 15 examples" printed among tens of
    # thousands of progress-bar lines does not distinguish that from "13 of 15 accepted",
    # which is a normal, healthy run. Warn at WARNING and name the command, so the one
    # case that degrades the shipped model is greppable (bd fix-k0i.37).
    if not dict_examples and num_examples > 0:
        logger.warning(
            f"NO usable DSPy parameter examples for command '{command_name}': "
            f"{num_examples} requested, {len(examples)} parsed out of the response, "
            f"all {len(rejected_examples)} rejected. Parameter extraction for this "
            f"command will run ZERO-SHOT. Nothing was cached, so the next training "
            f"run retries; if it keeps happening, inspect the rejected examples in "
            f"___command_info/{command_name}_param_labeled.json."
        )

    # A draw that parsed to nothing is degraded data -- a truncated response, a model
    # that ignored the output format, a provider hiccup. Caching it would make that
    # one bad minute permanent: the command would few-shot prompt with an empty
    # example set on every subsequent run, with no failure left in sight to explain
    # why. Same ruling as R6 makes for a fallen-back utterance set.
    if cache is not None and fingerprint is not None and dict_examples:
        cache.store(fingerprint, seed, dict_examples, rejected_examples)

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
    
    
