# wrp/server/runtime/settings/providers/builtin.py
from __future__ import annotations

import os
from typing import Optional, cast

from pydantic import AnyHttpUrl, Field, SecretStr

from .settings import ProviderSettings


class OpenAIProviderSettings(ProviderSettings):
    """
    Predefined provider: 'openai'
    """
    api_key: SecretStr | None = Field(
        default=None, description="OpenAI API key"
    )
    base_url: AnyHttpUrl = Field(
        default=cast(AnyHttpUrl, "https://api.openai.com/v1"),
        description="OpenAI API base URL",
    )
    organization: Optional[str] = Field(
        default=None, description="Optional OpenAI organization identifier"
    )


class AnthropicProviderSettings(ProviderSettings):
    """
    Predefined provider: 'anthropic'
    """
    api_key: SecretStr | None = Field(
        default=None, description="Anthropic API key"
    )
    base_url: AnyHttpUrl = Field(
        default=cast(AnyHttpUrl, "https://api.anthropic.com"),
        description="Anthropic API base URL",
    )


class GoogleProviderSettings(ProviderSettings):
    """
    Predefined provider: 'google' (Gemini)
    """
    api_key: SecretStr | None = Field(
        default=None, description="Google Gemini API key"
    )
    base_url: AnyHttpUrl = Field(
        default=cast(AnyHttpUrl, "https://generativelanguage.googleapis.com"),
        description="Gemini API base URL",
    )
    project: Optional[str] = Field(
        default=None, description="Optional Google Cloud project (if applicable)"
    )


class LiteLLMProviderSettings(ProviderSettings):
    """
    Predefined provider: 'litellm'
    Treat LiteLLM as a hosted/router API (or local gateway) requiring a key + base URL.
    """
    api_key: SecretStr | None = Field(
        default=None, description="LiteLLM API key (router access token)"
    )
    base_url: AnyHttpUrl = Field(
        default=cast(AnyHttpUrl, "https://api.litellm.ai"),
        description="LiteLLM router/base URL",
    )
