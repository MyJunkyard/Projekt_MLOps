"""
common/hashing.py — Hash helpers for run provenance.

Moved from ``train.py`` (Workstream 0 module restructure): the params
hash is provenance metadata, not training logic. WS0 moved
``get_git_commit_hash`` here as well so every stage (including
``features``, which must not import ``training``) can record provenance.
"""

import hashlib
import subprocess
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


def compute_sha256_file(path: str | Path) -> str:
    """SHA256 hex digest of a file's bytes (WS0 superset provenance).

    Args:
        path: File to hash (superset parquet, schema sidecar, raw CSV...).

    Returns:
        The SHA256 hex digest of the file contents.

    Raises:
        OSError: If the file cannot be read (callers guard on existence).
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_git_commit_hash() -> str:
    """Get the current git commit hash, or 'unknown' if not in a git repo.

    Moved here from ``training/loader.py`` (WS0): ``features`` records it
    in the superset manifest and must not import ``training``.
    ``training.loader.get_git_commit_hash`` re-exports this function.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
