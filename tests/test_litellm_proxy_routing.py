"""
Tests for LiteLLM Proxy routing functionality in dspy_utils.get_lm().

These tests verify the parameter resolution logic without making actual LLM calls.
"""

import pytest
import fastworkflow


class TestLiteLLMProxyRouting:
    """Tests for litellm_proxy/ model routing in get_lm()."""
    
    @pytest.fixture(autouse=True)
    def reset_env_vars(self):
        """Reset fastworkflow env vars before each test."""
        # Store original env vars
        original_env_vars = dict(fastworkflow._env_vars)
        yield
        # Restore original env vars
        fastworkflow._env_vars.clear()
        fastworkflow._env_vars.update(original_env_vars)
    
    def test_proxy_model_with_api_base_and_key(self):
        """
        When model starts with litellm_proxy/ and LITELLM_PROXY_API_BASE is set,
        the LM should be configured with api_base and optional api_key.
        """
        from fastworkflow.utils.dspy_utils import get_lm
        
        # Configure proxy settings
        fastworkflow._env_vars["LLM_TEST"] = "litellm_proxy/bedrock_mistral_large_2407"
        fastworkflow._env_vars["LITELLM_PROXY_API_BASE"] = "http://127.0.0.1:4000"
        fastworkflow._env_vars["LITELLM_PROXY_API_KEY"] = "test-proxy-key"
        
        # Create LM (api_key_env_var is ignored for proxy models)
        lm = get_lm("LLM_TEST", "SOME_IGNORED_KEY_VAR")
        
        # Verify the LM was created with proxy settings
        assert lm.model == "litellm_proxy/bedrock_mistral_large_2407"
        # DSPy LM stores kwargs that were passed to it
        assert lm.kwargs.get("api_base") == "http://127.0.0.1:4000"
        assert lm.kwargs.get("api_key") == "test-proxy-key"
    
    def test_proxy_model_without_api_key(self):
        """
        When model starts with litellm_proxy/ but LITELLM_PROXY_API_KEY is not set,
        the LM should still be created (for no-auth proxies).
        """
        from fastworkflow.utils.dspy_utils import get_lm
        
        # Configure proxy settings without API key
        fastworkflow._env_vars["LLM_TEST"] = "litellm_proxy/my_custom_model"
        fastworkflow._env_vars["LITELLM_PROXY_API_BASE"] = "http://localhost:8080"
        # LITELLM_PROXY_API_KEY is not set
        
        # Create LM
        lm = get_lm("LLM_TEST")
        
        # Verify the LM was created with api_base but no api_key
        assert lm.model == "litellm_proxy/my_custom_model"
        assert lm.kwargs.get("api_base") == "http://localhost:8080"
        assert "api_key" not in lm.kwargs or lm.kwargs.get("api_key") is None
    
    def test_proxy_model_missing_api_base_raises_error(self):
        """
        When model starts with litellm_proxy/ but LITELLM_PROXY_API_BASE is not set,
        a clear ValueError should be raised.
        """
        from fastworkflow.utils.dspy_utils import get_lm
        
        # Configure model but not the proxy base URL
        fastworkflow._env_vars["LLM_TEST"] = "litellm_proxy/some_model"
        # LITELLM_PROXY_API_BASE is not set
        
        # Should raise ValueError with helpful message
        with pytest.raises(ValueError) as exc_info:
            get_lm("LLM_TEST")
        
        error_message = str(exc_info.value)
        assert "litellm_proxy/" in error_message
        assert "LITELLM_PROXY_API_BASE" in error_message
    
    def test_direct_model_uses_role_specific_key(self):
        """
        Non-proxy models should use the role-specific API key (existing behavior).
        """
        from fastworkflow.utils.dspy_utils import get_lm
        
        # Configure direct provider model
        fastworkflow._env_vars["LLM_AGENT"] = "mistral/mistral-small-latest"
        fastworkflow._env_vars["LITELLM_API_KEY_AGENT"] = "direct-api-key"
        
        # Create LM with role-specific key
        lm = get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
        
        # Verify direct provider settings
        assert lm.model == "mistral/mistral-small-latest"
        assert lm.kwargs.get("api_key") == "direct-api-key"
        # api_base should not be set for direct models
        assert "api_base" not in lm.kwargs or lm.kwargs.get("api_base") is None
    
    def test_direct_model_without_key(self):
        """
        Non-proxy models without an API key should still work (for providers that don't need one).
        """
        from fastworkflow.utils.dspy_utils import get_lm
        
        # Configure model without key
        fastworkflow._env_vars["LLM_LOCAL"] = "ollama/llama2"
        
        # Create LM without api_key_env_var
        lm = get_lm("LLM_LOCAL")
        
        # Verify model was created
        assert lm.model == "ollama/llama2"
    
    def test_missing_model_env_var_raises_error(self):
        """
        When the model environment variable is not set, a clear error should be raised.
        """
        from fastworkflow.utils.dspy_utils import get_lm
        
        # LLM_NONEXISTENT is not set
        
        with pytest.raises(ValueError) as exc_info:
            get_lm("LLM_NONEXISTENT")
        
        error_message = str(exc_info.value)
        assert "LLM_NONEXISTENT" in error_message
    
    def test_proxy_model_preserves_kwargs(self):
        """
        Additional kwargs passed to get_lm() should be preserved for proxy models.
        """
        from fastworkflow.utils.dspy_utils import get_lm
        
        # Configure proxy settings
        fastworkflow._env_vars["LLM_TEST"] = "litellm_proxy/test_model"
        fastworkflow._env_vars["LITELLM_PROXY_API_BASE"] = "http://127.0.0.1:4000"
        fastworkflow._env_vars["LITELLM_PROXY_API_KEY"] = "proxy-key"
        
        # Create LM with additional kwargs
        lm = get_lm("LLM_TEST", temperature=0.7, max_tokens=1000)
        
        # Verify kwargs are preserved
        assert lm.kwargs.get("api_base") == "http://127.0.0.1:4000"
        assert lm.kwargs.get("api_key") == "proxy-key"
        assert lm.kwargs.get("temperature") == 0.7
        assert lm.kwargs.get("max_tokens") == 1000
    
    def test_direct_model_preserves_kwargs(self):
        """
        Additional kwargs passed to get_lm() should be preserved for direct models.
        """
        from fastworkflow.utils.dspy_utils import get_lm
        
        # Configure direct model
        fastworkflow._env_vars["LLM_TEST"] = "openai/gpt-4"
        fastworkflow._env_vars["LITELLM_API_KEY_TEST"] = "openai-key"
        
        # Create LM with additional kwargs
        lm = get_lm("LLM_TEST", "LITELLM_API_KEY_TEST", temperature=0.5)
        
        # Verify kwargs are preserved
        assert lm.kwargs.get("api_key") == "openai-key"
        assert lm.kwargs.get("temperature") == 0.5


class TestMixedModelConfiguration:
    """Tests for mixing proxy and direct models in the same application."""
    
    @pytest.fixture(autouse=True)
    def reset_env_vars(self):
        """Reset fastworkflow env vars before each test."""
        original_env_vars = dict(fastworkflow._env_vars)
        yield
        fastworkflow._env_vars.clear()
        fastworkflow._env_vars.update(original_env_vars)
    
    def test_mixed_proxy_and_direct_models(self):
        """
        Application can use both proxy and direct models simultaneously.
        """
        from fastworkflow.utils.dspy_utils import get_lm
        
        # Configure proxy for agent
        fastworkflow._env_vars["LLM_AGENT"] = "litellm_proxy/bedrock_model"
        fastworkflow._env_vars["LITELLM_PROXY_API_BASE"] = "http://proxy:4000"
        fastworkflow._env_vars["LITELLM_PROXY_API_KEY"] = "proxy-key"
        
        # Configure direct for param extraction
        fastworkflow._env_vars["LLM_PARAM_EXTRACTION"] = "mistral/mistral-small-latest"
        fastworkflow._env_vars["LITELLM_API_KEY_PARAM_EXTRACTION"] = "mistral-key"
        
        # Create both LMs
        agent_lm = get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")  # key ignored for proxy
        param_lm = get_lm("LLM_PARAM_EXTRACTION", "LITELLM_API_KEY_PARAM_EXTRACTION")
        
        # Verify agent uses proxy
        assert agent_lm.model == "litellm_proxy/bedrock_model"
        assert agent_lm.kwargs.get("api_base") == "http://proxy:4000"
        assert agent_lm.kwargs.get("api_key") == "proxy-key"
        
        # Verify param extraction uses direct
        assert param_lm.model == "mistral/mistral-small-latest"
        assert param_lm.kwargs.get("api_key") == "mistral-key"
        assert "api_base" not in param_lm.kwargs or param_lm.kwargs.get("api_base") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
