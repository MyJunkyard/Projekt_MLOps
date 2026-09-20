"""
features/main.py — Featurisation orchestration.

Orchestrates: load raw → calendar/holiday features → availability
alignment → weather merge → lag/rolling features → derivative features
→ split → save. The only place in the features package that knows the
run order.

Workstream 4 additions:

- **Availability alignment** (``features.availability_lags``): every
  configured raw external column (``load_mw``, ``{source}_mw``) is
  replaced by ``{col}_lag{L}h`` — the value actually known at
  prediction time. Actual load and generation are published with a
  ~1h real-time lag; merging the raw current-hour value would leak
  information that does not exist at serving time. The raw columns
  never reach ``features.parquet``.
- **Column metadata registry**: each block runner declares the columns
  it produced as ``ColumnSpec`` records (``src/common/schema.py``);
  ``build_features`` returns ``(df, FeatureSchema)`` and ``main()``
  saves ``features_schema.json`` next to ``features.parquet``.
  Downstream consumers query the schema instead of parsing name
  patterns.

Structure: ``build_features`` keeps the explicit ``if enabled`` dispatch
in run order (order is load-bearing for the no-leakage guarantee — kept
visible, no registry) and delegates each group to a ``_run_*`` block
runner. A runner applies its transformation(s) via ``_apply_step`` and
then declares every column it added via :func:`_declare_new_columns` —
specs are derived from the actual column diff, so a block can never
declare a column it failed to add (and the final drift guard catches
the inverse).
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.logsetup import setup_logging
from src.common.schema import ColumnRole, ColumnSpec, FeatureSchema
from src.common.splits import get_split_masks
from src.config import load_config
from src.config.models import DataConfig, FeaturesConfig
from src.features.calendar import (
    _get_holiday_dates,
    add_calendar_features,
    add_holiday_proximity_features,
)
from src.features.derivatives import add_derivative_features
from src.features.lags import add_lag_features, add_rolling_features
from src.features.naming import make_availability_lag_name
from src.ingestion.weather import load_weather_cache, merge_weather

# Stable module name (not `__name__` — under `python -m` it becomes
# `"__main__"` and would bypass the configured src logger).
MODULE_LOGGER_NAME = "src.features.main"
logger = logging.getLogger(MODULE_LOGGER_NAME)

# Bookkeeping columns with dedicated roles (declared by _base_specs, so
# block runners must never re-declare them).
_META_COLUMNS = frozenset({"timestamp", "is_imputed"})


def _apply_step(
    df: pd.DataFrame, label: str, fn: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """Run one feature-group step with uniform logging.

    Standard block mechanics for every group in ``build_features``:
    INFO-log the step label, apply ``fn``, DEBUG-log exactly the columns
    it added. Spec declaration happens in the block runner (via
    :func:`_declare_new_columns`), keeping this helper mechanics-only.

    Args:
        df: Input frame.
        label: Human-readable step description, INFO-logged before running.
        fn: Pure transformation applied to ``df``.

    Returns:
        The transformed DataFrame.
    """
    logger.info("Adding %s", label)
    before = set(df.columns)
    df = fn(df)
    logger.debug("Added %s: %s", label, sorted(set(df.columns) - before))
    return df


def _declare_new_columns(
    df: pd.DataFrame,
    declared: dict[str, ColumnSpec],
    group: str,
    description: str,
    **extra: Any,
) -> list[ColumnSpec]:
    """Declare specs for the frame's not-yet-declared columns.

    The spec-declaration primitive: after a block runner applied its
    transformation, this declares every column the block added (columns
    already in ``declared`` — base, meta, or earlier blocks — are
    skipped and left untouched). Because declaration is driven by the
    actual frame contents, a block can never declare a column it failed
    to add; the final drift guard in ``build_features`` catches the
    inverse (added-but-undeclared is impossible by construction,
    dropped-but-still-declared is popped by the runner).

    Args:
        df: The frame *after* the block ran.
        declared: The running name→spec registry (mutated: new columns
            are added).
        group: Feature family label for the new columns.
        description: Human-readable description for the new columns.
        **extra: Extra ``ColumnSpec`` fields (e.g.
            ``availability_lag_hours``, ``derived_from``).

    Returns:
        The specs declared by this call (also added to ``declared``).
    """
    new: list[ColumnSpec] = []
    for col in df.columns:
        if col in declared:
            continue
        spec = ColumnSpec(
            name=col,
            role=ColumnRole.FEATURE,
            group=group,
            dtype=str(df[col].dtype),
            description=description,
            **extra,
        )
        declared[col] = spec
        new.append(spec)
    return new


def align_availability(
    df: pd.DataFrame, raw_col: str, lag_hours: int
) -> pd.DataFrame:
    """Replace a raw external column with its availability-lagged form.

    Adds ``{raw_col}_lag{L}h`` = ``df[raw_col].shift(L)`` and **drops**
    ``raw_col``, so the current-hour value (unknown at prediction time)
    can never become a feature. The first ``L`` rows are NaN and are
    removed by the final ``dropna()`` in ``build_features``.

    No-leakage contract: the shift is past-only and the raw column is
    removed in the same step — pinned by
    ``tests/unit/features/test_availability.py``.

    Args:
        df: DataFrame containing ``raw_col``, sorted by ``timestamp``
            (the shift is order-dependent).
        raw_col: Raw external column name (e.g. ``load_mw``, ``wind_mw``).
        lag_hours: Real-time publication lag in hours (≥ 1).

    Returns:
        The same DataFrame (mutated in place and returned) with the raw
        column replaced by the lagged one.

    Raises:
        ValueError: If ``lag_hours`` < 1 (via
            ``features.naming.make_availability_lag_name``).
    """
    lagged_col = make_availability_lag_name(raw_col, lag_hours)
    df[lagged_col] = df[raw_col].shift(lag_hours)
    return df.drop(columns=[raw_col])


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV and parse the timestamp column.

    Input contract: the CSV must contain a ``timestamp`` column
    (produced by the ingest stage, alongside ``price_eur_mwh`` and
    optionally ``load_mw`` / ``is_imputed``).

    Args:
        path: Path to the raw CSV file.

    Returns:
        A DataFrame with ``timestamp`` parsed as datetime (tz-naive —
        pandas parses ISO UTC timestamps to naive datetimes here) and
        the remaining columns as written by ingest.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def train_val_test_split(
    df: pd.DataFrame, data: DataConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets based on date boundaries in config.

    Args:
        df: DataFrame with a tz-aware UTC ``timestamp`` column (the
            featurisation output contract: ``timestamp``, target column,
            and all engineered feature columns).
        data: ``DataConfig`` with the ``train_end`` and ``val_end``
            boundaries.

    Returns:
        A tuple ``(train_df, val_df, test_df)`` of disjoint DataFrames
        with the same columns as the input.
    """
    train_mask, val_mask, test_mask = get_split_masks(df, data)

    train = df[train_mask].copy()
    val = df[val_mask].copy()
    test = df[test_mask].copy()

    logger.info(
        "Train split: %s rows (%s to %s)",
        f"{len(train):,}",
        train["timestamp"].min(),
        train["timestamp"].max(),
    )
    logger.info(
        "Val split: %s rows (%s to %s)",
        f"{len(val):,}",
        val["timestamp"].min(),
        val["timestamp"].max(),
    )
    logger.info(
        "Test split: %s rows (%s to %s)",
        f"{len(test):,}",
        test["timestamp"].min(),
        test["timestamp"].max(),
    )

    return train, val, test


def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    processed_path: str,
    reference_path: str,
) -> None:
    """Save concatenated features and reference dataset.

    Args:
        train: Training split DataFrame (must share the same columns as
            ``val`` and ``test``).
        val: Validation split DataFrame.
        test: Test split DataFrame.
        processed_path: Path to save the concatenated features Parquet.
        reference_path: Path to save the train split as reference data.
    """
    processed_path_obj = Path(processed_path)
    processed_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate all splits
    full = pd.concat([train, val, test], axis=0)
    full.to_parquet(processed_path_obj, index=False)
    logger.info(
        "Saved features: %s (%s rows)",
        processed_path_obj,
        f"{len(full):,}",
    )

    # Save train split as reference for drift detection
    reference_path_obj = Path(reference_path)
    reference_path_obj.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(reference_path_obj, index=False)
    logger.info(
        "Saved reference dataset: %s (%s rows)",
        reference_path_obj,
        f"{len(train):,}",
    )


# ---------------------------------------------------------------------------
# Base-column declaration
# ---------------------------------------------------------------------------


def _base_specs(df: pd.DataFrame, target_col: str) -> dict[str, ColumnSpec]:
    """Declare the non-engineered columns present after ingest.

    Args:
        df: The frame *before* any feature block ran (base columns only).
        target_col: Name of the target column.

    Returns:
        The initial name→spec registry: ``timestamp`` (identifier), the
        target, ``is_imputed`` (meta, when present), and every remaining
        raw external column (feature, group ``external``).
    """
    declared: dict[str, ColumnSpec] = {}
    for col in df.columns:
        if col == "timestamp":
            declared[col] = ColumnSpec(
                name=col,
                role=ColumnRole.IDENTIFIER,
                group="base",
                dtype="datetime64[ns, UTC]",
            )
        elif col == target_col:
            declared[col] = ColumnSpec(
                name=col, role=ColumnRole.TARGET, group="base"
            )
        elif col == "is_imputed":
            declared[col] = ColumnSpec(
                name=col, role=ColumnRole.META, group="imputation", dtype="bool"
            )
        else:
            declared[col] = ColumnSpec(
                name=col,
                role=ColumnRole.FEATURE,
                group="external",
                description="raw external column from ingest",
            )
    return declared


# ---------------------------------------------------------------------------
# Block runners (one per feature group, in build_features run order)
# ---------------------------------------------------------------------------


def _run_calendar(
    df: pd.DataFrame, declared: dict[str, ColumnSpec]
) -> pd.DataFrame:
    """Calendar + holiday-proximity block (group ``calendar``)."""
    # Build the holiday calendar once for the data's year range (padded
    # by one year on each side) and share it across all holiday features
    holidays = _get_holiday_dates(df)
    df = _apply_step(
        df,
        "calendar features",
        lambda d: add_calendar_features(d, holidays),
    )
    _declare_new_columns(df, declared, "calendar", "calendar/holiday feature")
    df = _apply_step(
        df,
        "holiday proximity features",
        lambda d: add_holiday_proximity_features(d, holidays),
    )
    _declare_new_columns(
        df, declared, "calendar", "holiday proximity feature"
    )
    return df


def _run_availability(
    df: pd.DataFrame,
    features: FeaturesConfig,
    declared: dict[str, ColumnSpec],
) -> pd.DataFrame:
    """Availability-alignment block (WS4, group ``availability_lag``).

    Replaces each configured raw external column with ``{col}_lag{L}h``
    BEFORE the lag block, so target lags/rollings are computed on the
    aligned frame. Columns configured but absent from the frame (e.g.
    ``load_mw`` with ``include_load: false``, generation sources not
    downloaded) are skipped with a DEBUG log.
    """
    for raw_col in sorted(features.availability_lags):
        lag_hours = features.availability_lags[raw_col]
        if raw_col not in df.columns:
            logger.debug(
                "availability_lags: %s not present in frame — skipping",
                raw_col,
            )
            continue
        lagged_col = make_availability_lag_name(raw_col, lag_hours)
        df = _apply_step(
            df,
            f"availability alignment for {raw_col} (lag {lag_hours}h)",
            lambda d, col=raw_col, lag=lag_hours: align_availability(
                d, col, lag
            ),
        )
        # The raw column no longer exists — its base declaration must go,
        # or the final drift guard would flag a declared-but-absent column.
        declared.pop(raw_col, None)
        declared[lagged_col] = ColumnSpec(
            name=lagged_col,
            role=ColumnRole.FEATURE,
            group="availability_lag",
            dtype="float64",
            availability_lag_hours=lag_hours,
            derived_from=[raw_col],
            description=(
                f"{raw_col} as known {lag_hours}h before the row timestamp "
                "(real-time publication lag)"
            ),
        )
    return df


def _run_weather(
    df: pd.DataFrame,
    features: FeaturesConfig,
    weather: dict[str, pd.DataFrame],
    declared: dict[str, ColumnSpec],
) -> pd.DataFrame:
    """Weather-merge block (WS3, group ``weather``), one step per location."""
    for location in features.weather.locations:
        if location not in weather:
            raise ValueError(
                f"features.weather.locations includes {location!r} but "
                "no weather frame was supplied for it — check the "
                "cache load in features.main()."
            )
        df = _apply_step(
            df,
            f"weather features for {location}",
            lambda d, loc=location, frame=weather[location]: merge_weather(
                d, frame, loc
            ),
        )
        _declare_new_columns(
            df, declared, "weather", f"weather variable for {location}"
        )
    return df


def _run_lags(
    df: pd.DataFrame,
    features: FeaturesConfig,
    target_col: str,
    declared: dict[str, ColumnSpec],
) -> pd.DataFrame:
    """Target lag + rolling block (groups ``lag`` / ``rolling``)."""
    periods = features.lags.periods
    windows = features.lags.rolling_windows
    df = _apply_step(
        df,
        f"lag features for periods: {periods}",
        lambda d: add_lag_features(d, target_col, periods),
    )
    _declare_new_columns(df, declared, "lag", "target lag feature")
    df = _apply_step(
        df,
        f"rolling features for windows: {windows}",
        lambda d: add_rolling_features(d, target_col, windows),
    )
    _declare_new_columns(
        df, declared, "rolling", "trailing rolling statistic of the target"
    )
    return df


def _run_derivatives(
    df: pd.DataFrame,
    features: FeaturesConfig,
    target_col: str,
    declared: dict[str, ColumnSpec],
) -> pd.DataFrame:
    """Derivative block (group ``derivative``)."""
    order = features.derivatives.order
    smooth_window = features.derivatives.smooth_window
    df = _apply_step(
        df,
        f"derivative features (order={order}, smooth_window={smooth_window})",
        lambda d: add_derivative_features(
            d, target_col, order=order, smooth_window=smooth_window
        ),
    )
    _declare_new_columns(df, declared, "derivative", "smoothed target derivative")
    return df


def _drop_nan_rows(df: pd.DataFrame, features: FeaturesConfig) -> pd.DataFrame:
    """Drop NaN rows from lag/derivative creation; fail on an empty result.

    Raises:
        ValueError: If every row is dropped (e.g. the longest configured
            lag period exceeds the available history) — with an
            actionable hint when lags are enabled.
    """
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(
            "Dropped %d rows with NaN from lag/derivative creation", dropped
        )
    if df.empty:
        longest_lag = max(features.lags.periods) if features.lags.enabled else None
        hint = (
            f" The longest configured lag period is {longest_lag}h, which "
            "exceeds the available history — shorten "
            "features.lags.periods or provide more raw data."
            if longest_lag is not None
            else ""
        )
        raise ValueError(
            "Feature engineering produced an empty DataFrame "
            f"(dropped all {before} rows as NaN from lag/derivative "
            f"creation).{hint}"
        )
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    features: FeaturesConfig,
    target_col: str,
    weather: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, FeatureSchema]:
    """Apply all enabled feature groups to the full frame (pre-split).

    Pure transformation (no I/O): sorts by ``timestamp``, applies each
    enabled feature group in run order, and drops rows left as NaN by
    lag/derivative creation at the series start. The caller splits the
    returned frame into train/val/test.

    Run order (load-bearing for the no-leakage guarantee — kept visible,
    no registry):

    1. calendar/holiday features
    2. **availability alignment** (WS4): raw externals → ``{col}_lag{L}h``
    3. weather merge (WS3)
    4. target lag + rolling features
    5. derivative features

    Lag/rolling/availability features are computed here on the **full
    frame before splitting** — see ``features.lags`` and
    ``TestNoLeakageAcrossSplits``.

    Each block runner declares the columns it produced; the returned
    :class:`FeatureSchema` is the metadata counterpart of the frame
    (pinned equal to ``df.columns`` before return — drift fails fast).

    Args:
        df: Raw DataFrame with ``timestamp`` and ``target_col`` columns.
        features: ``FeaturesConfig`` with the per-group toggles/settings.
        target_col: Name of the target column to derive lags, rolling
            statistics, and derivatives from.
        weather: Optional per-location weather frames (``timestamp`` +
            float variable columns), keyed by location name. Required
            when ``features.weather.enabled`` is true; the caller loads
            them from the cache (never the network — decision D2).

    Returns:
        A ``(df, schema)`` tuple: the featurised DataFrame with no NaN
        rows, and the :class:`FeatureSchema` describing its columns.

    Raises:
        ValueError: If ``features.weather.enabled`` is true but no
            weather data was supplied, if a configured location is
            missing from ``weather``, if the schema and the frame
            disagree on columns, or if every row is dropped as NaN
            (e.g. the longest configured lag period exceeds the
            available history).
    """
    # Sort by timestamp to ensure correct lag computation
    df = df.sort_values("timestamp").reset_index(drop=True)
    declared = _base_specs(df, target_col)

    if features.calendar.enabled:
        df = _run_calendar(df, declared)

    if features.availability_lags:
        df = _run_availability(df, features, declared)

    if features.weather.enabled:
        if not weather:
            raise ValueError(
                "features.weather.enabled is true but no weather data was "
                "supplied — load the weather cache in features.main() and "
                "pass it as `weather`."
            )
        df = _run_weather(df, features, weather, declared)

    if features.lags.enabled:
        df = _run_lags(df, features, target_col, declared)

    if features.derivatives.enabled:
        df = _run_derivatives(df, features, target_col, declared)

    df = _drop_nan_rows(df, features)

    schema = FeatureSchema(columns=declared)
    # Drift guard: the declared schema must cover exactly the frame
    # columns — a block that drops a column without popping its
    # declaration (or otherwise diverges) fails here, not downstream.
    schema.assert_matches_dataframe(df)
    return df, schema


def main():
    """Orchestrate feature engineering pipeline."""
    cfg = load_config()
    # Configure the *package* logger by explicit name (never `__name__` —
    # under `python -m` that resolves to "__main__" and would bypass the
    # configured handlers; see MODULE_LOGGER_NAME above). Package scope so
    # every sibling module logger (calendar, lags, derivatives, ...)
    # inherits the handlers and level; leaf-scope would leave them
    # unconfigured (effective WARNING, INFO logs silently dropped).
    setup_logging(cfg.logging, logger_name="src.features")
    raw_path = Path(cfg.data.raw_path) / "entsoe_prices.csv"
    processed_path = cfg.data.processed_path
    reference_path = cfg.data.reference_path
    schema_path = Path(processed_path).with_name("features_schema.json")

    logger.info("Stage: featurisation")
    logger.info("Loading raw data")
    df = load_raw_data(str(raw_path))
    logger.info("Loaded %s rows from %s", f"{len(df):,}", raw_path)
    logger.debug("Raw data columns: %s", list(df.columns))

    # Weather (Workstream 3): the ingest stage owns all network I/O and
    # persists the per-location cache (decision D2); featurise only reads
    # it — never the network.
    weather: dict[str, pd.DataFrame] | None = None
    if cfg.features.weather.enabled:
        logger.info(
            "Loading weather cache for locations: %s",
            cfg.features.weather.locations,
        )
        weather = {
            location: load_weather_cache(
                cfg.data.raw_path,
                location,
                df["timestamp"].min(),
                df["timestamp"].max(),
                cfg.features.weather.variables,
                allow_synthetic=cfg.features.weather.allow_synthetic,
            )
            for location in cfg.features.weather.locations
        }
    else:
        logger.debug("Weather features disabled — no weather merge")

    logger.info("Engineering features")
    df, schema = build_features(
        df, cfg.features, cfg.data.target_col, weather=weather
    )

    logger.debug("Feature columns after engineering: %s", list(df.columns))
    logger.info("Splitting into train/val/test")
    train, val, test = train_val_test_split(df, cfg.data)

    logger.info("Saving processed data")
    save_processed_data(train, val, test, processed_path, reference_path)
    schema.save(schema_path)
    logger.info("Featurisation complete")


if __name__ == "__main__":
    main()
