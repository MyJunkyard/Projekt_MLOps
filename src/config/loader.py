"""
config/loader.py — Load and validate ``params.yaml`` into ``PipelineConfig``.

The only place that reads the YAML file. Validation happens here, at
load time — a malformed or inconsistent config fails before any pipeline
stage runs. ``FileNotFoundError`` and ``yaml.YAMLError`` propagate
unchanged; schema problems raise ``pydantic.ValidationError``.
"""

import yaml

from src.config.models import PipelineConfig


def load_config(config_path: str = "params.yaml") -> PipelineConfig:
    """Load pipeline configuration from a YAML file and validate it.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The validated ``PipelineConfig`` model.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not parseable YAML.
        pydantic.ValidationError: If the config violates the schema
            (missing required key, wrong type, unknown key, or a
            cross-field inconsistency such as ``val_end <= train_end``).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig.model_validate(raw)
