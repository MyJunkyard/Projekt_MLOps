"""
config.py — Pipeline configuration loading.

Interim home of ``load_config`` (moved from ``utils.py`` in the
Workstream 0 module restructure). Workstream 1 replaces the body with a
validated pydantic ``PipelineConfig`` model; until then the raw-dict
behaviour is unchanged so call sites keep working.
"""

import yaml


def load_config(config_path: str = "params.yaml") -> dict:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary of configuration values.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
