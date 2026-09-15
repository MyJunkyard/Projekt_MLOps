"""
common/hashing.py — Hash helpers for run provenance.

Moved from ``train.py`` (Workstream 0 module restructure): the params
hash is provenance metadata, not training logic.
"""

import hashlib
from pathlib import Path


def compute_params_hash(config_path: str = "params.yaml") -> str:
    """Compute SHA256 hash of the params.yaml file.

    The hash serves as a quick fingerprint for comparing runs — two runs
    with the same hash used identical configuration. For full reproducibility,
    the params.yaml file itself is also logged as an MLflow artifact (see
    ``training.registry.log_to_mlflow``).

    Args:
        config_path: Path to the params.yaml file.

    Returns:
        The SHA256 hex digest of the file contents, or "unknown" if the
        file does not exist.
    """
    path_obj = Path(config_path)
    if not path_obj.exists():
        return "unknown"
    return hashlib.sha256(path_obj.read_bytes()).hexdigest()
