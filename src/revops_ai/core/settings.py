"""Engine configuration models.

Credentials never live in code: connector settings are pydantic-settings
models that read from the environment, and the LLM is configured by id string
(``"provider:model"``, pydantic-ai's format) with an optional fallback.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """Which model LLM-backed agents run on.

    ``model`` accepts a pydantic-ai model id (``"openai:gpt-4o"``,
    ``"anthropic:claude-sonnet-4-6"``) or, in tests, a model instance such as
    ``pydantic_ai.models.test.TestModel``.
    """

    model_config = {"arbitrary_types_allowed": True}

    model: str | object
    fallback: str | None = None


class FreshnessPolicy(BaseModel):
    """How stale connector data may be before a run is annotated or aborted."""

    max_staleness: timedelta
    on_violation: Literal["warn", "fail"] = "warn"


class EngineSettings(BaseSettings):
    """Environment-driven engine defaults (REVOPS_AI_* variables)."""

    model_config = SettingsConfigDict(env_prefix="REVOPS_AI_")

    llm_model: str | None = None
    llm_fallback: str | None = None
