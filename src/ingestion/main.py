"""
ingestion/main.py — Ingest orchestration.

Orchestrates: download → validate → fill → drop → save → manifest.
The only place in the ingestion package that knows the run order.
Moved verbatim from ``ingest.py`` (Workstream 0 module restructure).
"""

import logging
from pathlib import Path

from src.common.logsetup import setup_logging
from src.config import load_config
from src.ingestion.entsoe import (
    download_entsoe_data,
    generate_synthetic_data,
    ingest_generation_mix,
)
from src.ingestion.manifest import (
    SOURCE_ENTSOE,
    SOURCE_SYNTHETIC,
    ensure_consistent_sources,
    save_raw_data,
    write_manifest,
)
from src.ingestion.validation import fill_gaps, validate_entsoe_data
from src.ingestion.weather import ingest_weather

# Stable module name (not `__name__` — under `python -m` it is `"__main__"`
# and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.ingestion.main"
logger = logging.getLogger(MODULE_LOGGER_NAME)


def main():
    """Orchestrate: download → validate → fill → drop → save → manifest."""
    cfg = load_config()
    # Configure the *package* logger by explicit name (never `__name__` —
    # under `python -m` that resolves to "__main__" and would bypass the
    # configured handlers; see MODULE_LOGGER_NAME above). Package scope so
    # every sibling module logger (entsoe, validation, manifest, ...)
    # inherits the handlers and level; leaf-scope would leave them
    # unconfigured (effective WARNING, INFO logs silently dropped).
    setup_logging(cfg.logging, logger_name="src.ingestion")
    raw_path = cfg.data.raw_path
    freq = cfg.temporal.pandas_freq

    logger.info("Stage: ingestion")

    # Try to download real data; fall back to synthetic if no API key.
    # The source is recorded in the manifest so runs are auditable and
    # the source-consistency gate below can reject real/synthetic mixes
    # (Workstream 3, decision D7).
    try:
        logger.info("Attempting ENTSO-E data download")
        df = download_entsoe_data(cfg.data.entsoe)
        logger.info("Downloaded %s rows from ENTSO-E", f"{len(df):,}")
        entsoe_source = SOURCE_ENTSOE
    except ValueError as e:
        logger.warning("ENTSO-E download unavailable: %s", e)
        logger.warning("Falling back to synthetic data generation")
        df = generate_synthetic_data(
            include_load=cfg.data.entsoe.include_load,
            include_generation=cfg.features.generation_mix.enabled,
            generation_sources=cfg.features.generation_mix.sources,
        )
        logger.info("Generated %s synthetic rows", f"{len(df):,}")
        entsoe_source = SOURCE_SYNTHETIC

    # Guard against empty data (review point 4): a successful download can
    # still return zero rows (e.g. no data in the requested date range),
    # which would otherwise crash later in reindex_to_grid with an opaque
    # ValueError from df["timestamp"].min() on an empty series.
    if df.empty:
        logger.error(
            "Ingestion produced an empty DataFrame (no rows in the requested "
            "date range)"
        )
        raise ValueError(
            "Ingestion produced an empty DataFrame (no rows in the requested "
            "date range). Check data.entsoe.start_date / bidding_zone or "
            "API availability."
        )

    # Validate
    logger.debug("Data columns: %s", list(df.columns))
    logger.info("Validating data")
    validate_entsoe_data(df, cfg.data, cfg.temporal)
    logger.info("Validation passed")

    # Fill short gaps only (long gaps remain NaN — never ffilled).
    # fill_gaps reindexes to the config-derived grid internally (review point
    # 7: no hardcoded hourly assumption in main() — with temporal.resolution
    # set to daily/weekly, reindexing here at "h" would fabricate 23/167 NaN
    # rows per period that fill_gaps would then misclassify as long gaps).
    logger.info(
        "Filling gaps (max_gap_periods=%d, fill_method=%s, freq=%s)",
        cfg.data.max_gap_periods,
        cfg.data.fill_method,
        freq,
    )
    df, imputation_stats = fill_gaps(
        df,
        max_gap_periods=cfg.data.max_gap_periods,
        fill_method=cfg.data.fill_method,
        freq=freq,
        add_is_imputed_flag=cfg.data.add_is_imputed_flag,
    )

    # Explicitly drop (or retain) rows that remain NaN after imputation —
    # governed by the data.drop_long_gaps config flag (review points 1 and 1b).
    n_unfilled = imputation_stats["n_unfilled"]
    if cfg.data.drop_long_gaps:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        n_dropped = before - len(df)
        if n_dropped > 0:
            logger.warning(
                "Dropped %d rows with remaining NaN values "
                "(gaps longer than max_gap_periods=%d)",
                n_dropped,
                cfg.data.max_gap_periods,
            )
        # All unfilled rows were removed, so nothing remains unfilled
        imputation_stats["n_unfilled_rows"] = 0
    else:
        n_dropped = 0
        imputation_stats["n_unfilled_rows"] = n_unfilled
        if n_unfilled > 0:
            logger.info(
                "Retaining %d unfilled row(s) (drop_long_gaps=false); "
                "downstream stages may drop them.",
                n_unfilled,
            )
    imputation_stats["n_dropped_rows"] = n_dropped

    # Save
    output_path = Path(raw_path) / "entsoe_prices.csv"
    csv_hash = save_raw_data(df, str(output_path))

    # Write manifest — includes imputation/drop stats (review point 1b)
    # and the provenance source (Workstream 3, decision D7)
    write_manifest(
        raw_path,
        df,
        sha256_hash=csv_hash,
        imputation_stats=imputation_stats,
        source=entsoe_source,
    )

    # Generation mix (Workstream 4): download-or-cache-hit per-source
    # generation, then outer-join onto the main frame on timestamp (same
    # pattern as load_mw) so the gap-filling stage handles coverage gaps.
    # The raw {source}_mw columns are aligned to their availability lag
    # by the featurise stage — they never become features unlagged.
    run_sources = {"entsoe": entsoe_source}
    if cfg.features.generation_mix.enabled:
        logger.info(
            "Fetching generation mix for sources: %s",
            cfg.features.generation_mix.sources,
        )
        generation_df, generation_source = ingest_generation_mix(
            cfg.features.generation_mix,
            cfg.data.entsoe,
            raw_path,
            df["timestamp"].min(),
            df["timestamp"].max(),
        )
        df = df.merge(generation_df, on="timestamp", how="outer")
        run_sources["entsoe/generation"] = generation_source
    else:
        logger.debug("Generation mix disabled — skipping generation ingest")

    # Weather acquisition (Workstream 3, decisions D2/D3): ingest owns
    # all network I/O; featurise later reads the cache strictly offline.
    # The range comes from the frame just downloaded and validated, so
    # weather exactly covers the price data.
    if cfg.features.weather.enabled:
        logger.info(
            "Fetching weather for locations: %s", cfg.features.weather.locations
        )
        _, weather_sources = ingest_weather(
            cfg.features.weather,
            raw_path,
            df["timestamp"].min(),
            df["timestamp"].max(),
        )
        run_sources.update(
            {f"weather/{loc}": src for loc, src in weather_sources.items()}
        )
    else:
        logger.debug("Weather features disabled — skipping weather ingest")

    # Source-consistency gate (decision D7): all-synthetic runs are a
    # valid offline mode; any real+synthetic mix poisons training and is
    # rejected outright.
    ensure_consistent_sources(run_sources)

    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
