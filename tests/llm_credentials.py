"""Shared gate for tests that need a working LLM_SYNDATA_GEN model."""


def _looks_like_real_key(value) -> bool:
    """Reject empty / placeholder keys like ``<API KEY ...>``."""
    return bool(value) and "<" not in value and "your-" not in value.lower()


def syndata_llm_available(env_vars) -> bool:
    """True when LLM_SYNDATA_GEN can authenticate.

    Either a real LITELLM_API_KEY_SYNDATA_GEN is present, or the model is a
    ``bedrock/`` model, which authenticates from the ambient AWS credential chain
    and must be left without a key (a non-empty key is sent to Bedrock as a bearer token).
    """
    if _looks_like_real_key(env_vars.get("LITELLM_API_KEY_SYNDATA_GEN")):
        return True
    return str(env_vars.get("LLM_SYNDATA_GEN") or "").startswith("bedrock/")
