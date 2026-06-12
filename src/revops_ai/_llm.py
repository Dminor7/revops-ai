"""Internal: resolve an LLMConfig to a pydantic-ai model.

This module (plus agents/base.py and core/router.py) is where pydantic-ai is
allowed to be imported. Nothing in the public API surface exposes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from revops_ai.core.settings import LLMConfig

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def resolve_model(config: LLMConfig) -> Model | str:
    from pydantic_ai.models import Model

    if config.fallback is not None and isinstance(config.model, str):
        from pydantic_ai.models.fallback import FallbackModel

        return FallbackModel(config.model, config.fallback)
    model = config.model
    if not isinstance(model, str | Model):
        raise TypeError(
            f"LLMConfig.model must be a model id string or a pydantic-ai Model, "
            f"got {type(model).__name__}."
        )
    return model
