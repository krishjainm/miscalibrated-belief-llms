"""
Registered model presets: Mistral, Qwen, OpenAI, Anthropic, Google.

Each preset maps to provider + model_id + optional base_url + env var for API key.
"""

from dataclasses import dataclass
from typing import Literal

Provider = Literal["openai", "anthropic", "google", "openai_compatible"]


@dataclass(frozen=True)
class ModelPreset:
    """Configuration for one named model endpoint."""

    name: str
    provider: Provider
    model_id: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    description: str = ""


PRESETS: dict[str, ModelPreset] = {
    "gpt-4o-mini": ModelPreset(
        name="gpt-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        description="OpenAI GPT-4o mini",
    ),
    "gpt-4o": ModelPreset(
        name="gpt-4o",
        provider="openai",
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        description="OpenAI GPT-4o",
    ),
    "claude-3-5-sonnet": ModelPreset(
        name="claude-3-5-sonnet",
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        api_key_env="ANTHROPIC_API_KEY",
        description="Anthropic Claude 3.5 Sonnet",
    ),
    "claude-3-5-haiku": ModelPreset(
        name="claude-3-5-haiku",
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        api_key_env="ANTHROPIC_API_KEY",
        description="Anthropic Claude 3.5 Haiku",
    ),
    "gemini-2.0-flash": ModelPreset(
        name="gemini-2.0-flash",
        provider="google",
        model_id="gemini-2.0-flash",
        api_key_env="GOOGLE_API_KEY",
        description="Google Gemini 2.0 Flash",
    ),
    "gemini-1.5-flash": ModelPreset(
        name="gemini-1.5-flash",
        provider="google",
        model_id="gemini-1.5-flash",
        api_key_env="GOOGLE_API_KEY",
        description="Google Gemini 1.5 Flash",
    ),
    "mistral-small": ModelPreset(
        name="mistral-small",
        provider="openai_compatible",
        model_id="mistral-small-latest",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        description="Mistral Small (latest)",
    ),
    "mistral-large": ModelPreset(
        name="mistral-large",
        provider="openai_compatible",
        model_id="mistral-large-latest",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        description="Mistral Large (latest)",
    ),
    "qwen2.5-72b-together": ModelPreset(
        name="qwen2.5-72b-together",
        provider="openai_compatible",
        model_id="Qwen/Qwen2.5-72B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        description="Qwen 2.5 72B Instruct on Together",
    ),
    "qwen2.5-7b-together": ModelPreset(
        name="qwen2.5-7b-together",
        provider="openai_compatible",
        model_id="Qwen/Qwen2.5-7B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        description="Qwen 2.5 7B Instruct on Together",
    ),
    "qwen2.5-72b-dashscope-intl": ModelPreset(
        name="qwen2.5-72b-dashscope-intl",
        provider="openai_compatible",
        model_id="qwen2.5-72b-instruct",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        description="Qwen2.5-72B Instruct via DashScope (Singapore / intl compatible-mode)",
    ),
    "qwen2.5-72b-dashscope-cn": ModelPreset(
        name="qwen2.5-72b-dashscope-cn",
        provider="openai_compatible",
        model_id="qwen2.5-72b-instruct",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        description="Qwen2.5-72B Instruct via DashScope (China Beijing compatible-mode)",
    ),
    "qwen2.5-7b-dashscope-intl": ModelPreset(
        name="qwen2.5-7b-dashscope-intl",
        provider="openai_compatible",
        model_id="qwen2.5-7b-instruct",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        description="Qwen2.5-7B Instruct via DashScope (intl compatible-mode)",
    ),
}


def list_presets() -> list[str]:
    return sorted(PRESETS.keys())


def resolve_preset(name: str) -> ModelPreset:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown model preset {name!r}. Choose one of: {', '.join(list_presets())}"
        )
    return PRESETS[name]
