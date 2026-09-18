"""
config — Typed pipeline configuration (pydantic schema + YAML loader).

Public API:
- ``load_config(path="params.yaml") -> PipelineConfig`` — read + validate.
- ``PipelineConfig`` and the section models (``DataConfig``,
  ``ModelConfig``, ...) — the validated configuration shape.

``models.py`` is a strict leaf (imports nothing from ``src``); every
other package may import from here without cycle risk.
"""

from src.config.loader import load_config
from src.config.models import (
    CalendarConfig,
    CrossBorderFlowsConfig,
    DataConfig,
    DerivativesConfig,
    EntsoeConfig,
    EvaluationConfig,
    FeaturesConfig,
    GenerationMixConfig,
    LagsConfig,
    LoggingConfig,
    MlflowConfig,
    ModelConfig,
    PipelineConfig,
    ServingConfig,
    TemporalConfig,
    WeatherConfig,
)

__all__ = [
    "CalendarConfig",
    "CrossBorderFlowsConfig",
    "DataConfig",
    "DerivativesConfig",
    "EntsoeConfig",
    "EvaluationConfig",
    "FeaturesConfig",
    "GenerationMixConfig",
    "LagsConfig",
    "LoggingConfig",
    "MlflowConfig",
    "ModelConfig",
    "PipelineConfig",
    "ServingConfig",
    "TemporalConfig",
    "WeatherConfig",
    "load_config",
]
