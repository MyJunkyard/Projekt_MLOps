"""
ingestion/manifest.py — Raw-data persistence and provenance manifest.

Extracted verbatim from ``ingest.py`` (Workstream 0 module restructure).
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODULE_LOGGER_NAME = "src.ingestion.manifest"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def write_manifest(
    raw_path: str,
    df: pd.DataFrame,
    sha256_hash: str | None = None,
    imputation_stats: dict | None = None,
) -> None:
    """Write a data manifest JSON file with download metadata.

    The manifest records the integrity hash, row counts, and — when
    ``imputation_stats`` is provided — how many rows were imputed vs.
    dropped, so that every run's data-quality story is auditable
    (see review point 1b). If ``sha256_hash`` is supplied it is used
    directly (this is the on-disk file hash from ``save_raw_data``),
    so the manifest and the printed summary reference the same file.

    Args:
        raw_path: Path to the raw data directory.
        df: The raw DataFrame that was saved. Must have a ``timestamp``
            column (used for the ``date_range`` manifest entry).
        sha256_hash: SHA256 of the saved CSV file. Computed from the
            DataFrame if not provided.
        imputation_stats: Optional dict with imputation statistics
            (``n_imputed``, ``n_unfilled``, ``max_gap_periods``,
            ``fill_method``, ``freq``).
    """
    path_obj = Path(raw_path)
    path_obj.mkdir(parents=True, exist_ok=True)

    if sha256_hash is None:
        # Compute SHA256 of the DataFrame content
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        sha256_hash = hashlib.sha256(csv_bytes).hexdigest()

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "date_range": [
            df["timestamp"].min().isoformat(),
            df["timestamp"].max().isoformat(),
        ],
        "row_count": len(df),
        "sha256": sha256_hash,
    }

    if imputation_stats:
        manifest["n_imputed_rows"] = imputation_stats.get("n_imputed", 0)
        manifest["n_dropped_rows"] = imputation_stats.get("n_dropped_rows", 0)
        # fill_gaps reports "n_unfilled"; main() may normalize it to
        # "n_unfilled_rows" after deciding what dropna() removes.
        manifest["n_unfilled_rows"] = imputation_stats.get(
            "n_unfilled_rows",
            imputation_stats.get("n_unfilled", 0),
        )
        manifest["max_gap_periods"] = imputation_stats.get("max_gap_periods", 2)
        manifest["fill_method"] = imputation_stats.get("fill_method", "ffill")
        manifest["freq"] = imputation_stats.get("freq", "h")

    manifest_path = path_obj / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)


def save_raw_data(df: pd.DataFrame, path: str) -> str:
    """Save raw data to CSV, log a summary, and return the on-disk SHA256.

    Args:
        df: DataFrame to save. Must have a ``timestamp`` column (used
            for the logged data-range summary).
        path: Target CSV path.

    Returns:
        The SHA256 hex digest of the on-disk file. Callers should pass this
        to ``write_manifest`` so the manifest and the logged summary come
        from the same source (fixes review point 9).
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path_obj, index=False)

    # Compute SHA256 of the on-disk file
    sha256_hash = hashlib.sha256(path_obj.read_bytes()).hexdigest()

    logger.info("Saved %s", path_obj)
    logger.info("Saved %s rows", f"{len(df):,}")
    logger.info(
        "Data range: %s to %s",
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    logger.info("SHA256: %s", sha256_hash)
    return sha256_hash
