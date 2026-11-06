# Before anything imports dspy: dspy 3.4 puts a lazy stand-in for `openai` in
# sys.modules, and litellm importing an openai submodule through that stand-in
# fails with a circular import. Loaded first, the real openai is used instead.
# Every module that imports dspy carries the litellm line directly above its dspy import.
# The dspy-first case (dspy imported before fastworkflow) is repaired in fastworkflow/__init__.py.
import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
from pydantic import BaseModel, Field
from typing import Type, Optional, Dict, Any, Union, get_args, get_origin, Tuple, List

import fastworkflow
from fastworkflow.utils.logging import logger


def get_lm(model_env_var: str, api_key_env_var: Optional[str] = None, **kwargs):
    """
    Get the dspy LM object.
    
    Supports LiteLLM Proxy routing: if the model string starts with 'litellm_proxy/',
    the call is routed through the LiteLLM Proxy using LITELLM_PROXY_API_BASE and
    LITELLM_PROXY_API_KEY environment variables.
    
    Args:
        model_env_var: Name of the environment variable containing the model string
                       (e.g., 'LLM_AGENT', 'LLM_PARAM_EXTRACTION').
        api_key_env_var: Name of the environment variable containing the API key
                         for direct provider calls. Ignored for litellm_proxy/ models.
        **kwargs: Additional keyword arguments passed to dspy.LM().
    
    Returns:
        dspy.LM: Configured language model instance.
    
    Raises:
        ValueError: If model is not set, or if using litellm_proxy/ without
                    LITELLM_PROXY_API_BASE configured.
    
    Example:
        # Direct provider call (existing behavior):
        # LLM_AGENT=mistral/mistral-small-latest
        # LITELLM_API_KEY_AGENT=sk-...
        lm = get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
        
        # LiteLLM Proxy call:
        # LLM_AGENT=litellm_proxy/bedrock_mistral_large_2407
        # LITELLM_PROXY_API_BASE=http://127.0.0.1:4000
        # LITELLM_PROXY_API_KEY=proxy-key-...
        lm = get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")  # api_key_env_var is ignored for proxy
    """
    # `FW_LM_CACHE=0` defeats the DSPy response cache process-wide
    # (`fix-bn1` `[XR16]`). Without it, k repeated attempts of one task in a
    # pass^k experiment send identical prompts, hit the disk+memory cache that
    # dspy enables by default, and come back byte-identical -- so pass^k equals
    # pass@1 by construction and the run looks like a spectacular result. The
    # lever is process-wide because nothing threads per-call kwargs down to
    # here; `**kwargs` already reaches `dspy.LM`, but no caller passes any.
    #
    # Unset leaves today's behavior exactly as it was. The value is the env file
    # ``init`` loaded, else the process environment (``get_env_var``, fix-hia1.40),
    # so a harness can set ``FW_LM_CACHE=0`` without calling ``init``.
    if fastworkflow.get_env_var("FW_LM_CACHE", default="1") in ("0", "false", "False"):
        kwargs.setdefault("cache", False)
    # Without one, a provider request that stalls blocks the turn for minutes.
    kwargs.setdefault(
        "timeout", float(fastworkflow.get_env_var("LLM_TIMEOUT_SECONDS", default="120")))

    model = fastworkflow.get_env_var(model_env_var)
    if not model:
        logger.critical(f"Critical Error: DSPy Language Model not provided. Set {model_env_var} environment variable.")
        raise ValueError(f"DSPy Language Model not provided. Set {model_env_var} environment variable.")
    
    # Check if this is a LiteLLM Proxy call
    if model.startswith("litellm_proxy/"):
        # Route through LiteLLM Proxy
        proxy_api_base = fastworkflow.get_env_var("LITELLM_PROXY_API_BASE")
        if not proxy_api_base:
            raise ValueError(
                f"Model '{model}' uses litellm_proxy/ prefix but LITELLM_PROXY_API_BASE is not set. "
                "Set LITELLM_PROXY_API_BASE to your LiteLLM Proxy URL (e.g., http://127.0.0.1:4000)."
            )
        
        # Get optional proxy API key (allows no-auth proxies when empty/not set)
        proxy_api_key = fastworkflow.get_env_var("LITELLM_PROXY_API_KEY", default=None)
        
        logger.debug(f"Routing {model_env_var} through LiteLLM Proxy at {proxy_api_base}")
        
        if proxy_api_key:
            return dspy.LM(
                model=model,
                api_base=proxy_api_base,
                api_key=proxy_api_key,
                **kwargs,
            )
        else:
            return dspy.LM(model=model, api_base=proxy_api_base, **kwargs)
    
    # Direct provider call (existing behavior)
    api_key = fastworkflow.get_env_var(api_key_env_var) if api_key_env_var else None
    return (
        dspy.LM(model=model, api_key=api_key, **kwargs)
        if api_key
        else dspy.LM(model=model, **kwargs)
    )

#: Models litellm refused a parameter for in this process (``params_rejected``). Their later calls
#: go with ``drop_params`` from the start (``call_dropping_rejected_params``).
_PARAMS_REJECTED: set[str] = set()


def params_rejected(error: BaseException) -> bool:
    """Whether ``error`` is litellm refusing, before it sends anything, a parameter the model does not take.

    That is ``litellm.UnsupportedParamsError`` anywhere in the cause chain (DSPy wraps it twice). Bedrock
    Claude Haiku 5.5, for one, takes only ``temperature=1``, so ``temperature=0.0`` is refused (fix-z62t).
    """
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, litellm.UnsupportedParamsError):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


def drops_rejected_params(lm) -> bool:
    """Whether calls on ``lm`` leave out the parameters its model does not take (so its temperature may not apply)."""
    return bool((getattr(lm, "kwargs", None) or {}).get("drop_params")) or getattr(lm, "model", None) in _PARAMS_REJECTED


def call_dropping_rejected_params(lm, call):
    """``call(lm)``; when litellm refuses one of ``lm``'s parameters, ``call`` once more with ``drop_params=True``.

    With ``drop_params`` litellm leaves out what the model does not take (a temperature other than 1 on a
    model that takes only 1) and sends the rest. The refusal is raised before anything is sent, so the
    first try costs no tokens. The model is remembered for the process: its later calls go with
    ``drop_params`` from the start, and the first refusal is logged once at WARNING.

    A model that takes the parameters is called exactly as before: ``drop_params`` is never added to
    it. Adding it to every LM is not neutral: DSPy sends an LM with ``drop_params`` through LiteLLM
    even when it would call the provider natively (``openai/``, ``anthropic/``). An LM litellm refused
    is already on LiteLLM.
    """
    dropping = bool((getattr(lm, "kwargs", None) or {}).get("drop_params"))
    if getattr(lm, "model", None) in _PARAMS_REJECTED and not dropping:
        return call(lm.copy(drop_params=True))
    try:
        return call(lm)
    except Exception as error:  # noqa: BLE001 - re-raised unless litellm refused a parameter
        if dropping or not params_rejected(error):
            raise
        if lm.model not in _PARAMS_REJECTED:
            _PARAMS_REJECTED.add(lm.model)
            reason = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
            logger.warning(f"{lm.model} refused a parameter ({reason}); its calls now leave out the "
                           "parameters it does not take")
    return call(lm.copy(drop_params=True))


def _process_field(field_info, is_input: bool) -> Tuple[Any, Any, bool]:
    """Process a single field and return its type, DSPy field, and optional status."""
    field_type = field_info.annotation
    field_desc = field_info.description or f"{'Input' if is_input else 'Output'} field"
    
    # Handle Optional types
    is_optional = False
    if get_origin(field_type) is Union:
        args = get_args(field_type)
        if type(None) in args:
            is_optional = True
            field_type = next((t for t in args if t is not type(None)), str)
    
    dspy_field = dspy.InputField(desc=field_desc) if is_input else dspy.OutputField(desc=field_desc)
    return field_type, dspy_field, is_optional


def _process_input_fields(model_class: Type[BaseModel], preserve_types: bool) -> Dict[str, Tuple]:
    """Process all input fields from a Pydantic model."""
    fields = {}
    for field_name, field_info in model_class.model_fields.items():
        field_type, dspy_field, _ = _process_field(field_info, is_input=True)
        if not preserve_types:
            field_type = str
        fields[field_name] = (field_type, dspy_field)
    return fields


def _process_output_fields(model_class: Type[BaseModel], preserve_types: bool) -> Tuple[Dict[str, Tuple], List[str]]:
    """Process all output fields from a Pydantic model and generate instructions."""
    fields = {}
    instructions = []
    
    for field_name, field_info in model_class.model_fields.items():
        field_type, dspy_field, _ = _process_field(field_info, is_input=False)
        if not preserve_types:
            field_type = str
        fields[field_name] = (field_type, dspy_field)
        
        # Generate field-specific instructions
        _add_field_instructions(field_name, field_info, instructions)
        
    return fields, instructions


def _add_field_instructions(field_name: str, field_info, instructions: List[str]) -> None:
    """Add instructions for a specific field based on its metadata."""
    if hasattr(field_info, 'default') and field_info.default is not None:
        instructions.append(f"For '{field_name}': Use '{field_info.default}' if not explicitly mentioned.")
    
    if hasattr(field_info, 'examples') and field_info.examples:
        examples_str = ", ".join(f"'{ex}'" for ex in field_info.examples)
        instructions.append(f"Examples for '{field_name}': {examples_str}")


def _create_instructions(custom_instructions: Optional[str], auto_instructions: List[str]) -> str:
    """Create the final instruction string from custom and auto-generated instructions."""
    if custom_instructions:
        return custom_instructions
    
    if not auto_instructions:
        return ""
        
    return "Extract the following fields based on the input:\n\n" + "\n".join(auto_instructions)


def dspySignature(
    Input_class: Type[BaseModel], 
    Output_class: Type[BaseModel],
    instructions: Optional[str] = None,
    preserve_types: bool = True
) -> Type[dspy.Signature]:
    """
    Dynamically creates a dspy.Signature class from Pydantic Input and Output models.

    Args:
        Input_class: A Pydantic BaseModel class defining the input fields.
        Output_class: A Pydantic BaseModel class defining the output fields.
        instructions: Optional custom instructions to include in the signature.
        preserve_types: Whether to preserve field type annotations in the signature.

    Returns:
        A new class that inherits from dspy.Signature.
    """
    if not issubclass(Input_class, BaseModel) or not issubclass(Output_class, BaseModel):
        raise TypeError("Input_class and Output_class must be subclasses of pydantic.BaseModel.")

    # Process fields from both classes
    input_fields = _process_input_fields(Input_class, preserve_types)
    output_fields, auto_instructions = _process_output_fields(Output_class, preserve_types)
    
    # Combine all fields and create instructions
    dspy_fields = {**input_fields, **output_fields}
    final_instructions = _create_instructions(instructions, auto_instructions)
    
    return dspy.Signature(dspy_fields, final_instructions.strip())

############################################
# Steps:
# 1. Define your signature and dspy function
# 2. Get prediction from DSPy module
# 3. Create output directly using ** unpacking

# dspy_signature_class = dspySignature(Signature.Input, Signature.Output)
# dspy_predict_func = dspy.Predict(dspy_signature_class)
# prediction = dspy_predict_func(input)  # Returns a dspy.Prediction object
# return Signature.Output(**prediction)