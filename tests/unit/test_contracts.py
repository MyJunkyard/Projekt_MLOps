"""
DataFrame-schema contract tests (Workstream 6a).

Executable pin of ``docs/contracts.md`` §3 — the stage-boundary
DataFrame schema. Each class runs one producer and asserts the output
contract, so a change to one stage that silently breaks a consumer
fails here first.

Boundaries covered:

1. ``TestIngestOutputContract`` — ``ingestion.main.main()`` output:
   ``data/raw/entsoe_prices.csv`` + ``manifest.json``.
2. ``TestFeaturiseOutputContract`` — ``features.main.build_features()``
   output: the features DataFrame + its ``FeatureSchema`` sidecar.
3. ``TestTrainInputContract`` — ``training.loader.load_features()``
   arrays vs ``get_feature_names`` / ``get_split_masks``.

All tests are fully offline: ENTSO-E downloads are mocked (or the API
key is blanked so the synthetic fallback runs), and weather frames are
built in memory — ``pytest`` never touches the network.
"""

import hashlib
import json
import logging
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.common.schema import ColumnRole
from src.common.splits import get_split_masks
from src.config.models import PipelineConfig
from src.features.main import (
    build_features,
    save_processed_data,
    train_val_test_split,
)
from src.ingestion.main import main as ingest_main
from src.training.loader import get_feature_names, load_features

# ---------------------------------------------------------------------------
# Contract vocabulary (docs/contracts.md §3)
# ---------------------------------------------------------------------------

CALENDAR_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "week_of_year",
    "is_holiday",
    "is_workday",
    "days_to_next_holiday",
    "days_since_last_holiday",
]
WEATHER_VARIABLES = ["temperature_2m", "wind_speed_100m"]
GENERATION_SOURCES = ["wind", "solar"]
LAG_PERIODS = [1, 24]
ROLLING_WINDOWS = [24]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    raw_path=None,
    include_load=True,
    add_is_imputed=True,
    allow_synthetic=False,
    **feature_sections,
):
    """Minimal valid PipelineConfig; every feature group defaults to off.

    ``feature_sections`` are merged into the ``features`` mapping (keys
    such as ``calendar``, ``lags``, ``derivatives``, ``weather``,
    ``generation_mix``, ``availability_lags``), so each test states
    exactly the flags whose contract it pins.
    """
    data = {
        "target_col": "price_eur_mwh",
        "train_end": "2023-06-05",
        "val_end": "2023-06-08",
        "add_is_imputed_flag": add_is_imputed,
        "entsoe": {
            "bidding_zone": "PSE",
            "start_date": "2023-06-01",
            "include_load": include_load,
            "allow_synthetic": allow_synthetic,
        },
    }
    if raw_path is not None:
        data["raw_path"] = str(raw_path)
    return PipelineConfig.model_validate(
        {
            "data": data,
            "features": feature_sections,
            "model": {"type": "sklearn.dummy.DummyRegressor"},
        }
    )


def _calendar_flags() -> dict[str, Any]:
    return {"calendar": {"enabled": True, "include": CALENDAR_COLUMNS}}


def _lags_flags() -> dict[str, Any]:
    return {
        "lags": {
            "enabled": True,
            "periods": LAG_PERIODS,
            "rolling_windows": ROLLING_WINDOWS,
        }
    }


def _weather_flags() -> dict[str, Any]:
    return {
        "weather": {
            "enabled": True,
            "variables": WEATHER_VARIABLES,
            "locations": ["warsaw"],
            "allow_synthetic": False,
        }
    }


def _generation_flags() -> dict[str, Any]:
    """Generation enabled plus the availability_lags its validator demands."""
    return {
        "generation_mix": {"enabled": True, "sources": GENERATION_SOURCES},
        "availability_lags": {
            "load_mw": 1,
            **{f"{source}_mw": 1 for source in GENERATION_SOURCES},
        },
    }


def _raw_df(
    n=400,
    start="2023-06-01",
    *,
    include_load=True,
    include_generation=False,
    include_imputed=True,
):
    """Frame matching the ingest-output contract (docs/contracts.md §3)."""
    timestamps = pd.date_range(start=start, periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_eur_mwh": np.linspace(40.0, 60.0, n),
        }
    )
    if include_load:
        df["load_mw"] = np.linspace(9_000.0, 11_000.0, n)
    if include_generation:
        df["wind_mw"] = np.linspace(500.0, 2_500.0, n)
        df["solar_mw"] = np.linspace(0.0, 1_500.0, n)
    if include_imputed:
        df["is_imputed"] = False
    return df


def _weather_frames(raw):
    """Synthetic tz-aware weather for ``warsaw`` over ``raw``'s timestamps."""
    n = len(raw)
    return {
        "warsaw": pd.DataFrame(
            {
                "timestamp": raw["timestamp"].reset_index(drop=True),
                "temperature_2m": np.linspace(-5.0, 25.0, n),
                "wind_speed_100m": np.linspace(1.0, 12.0, n),
            }
        )
    }


def _download_frame(include_load=True, n=96):
    """What the (mocked) price download returns: a contiguous hourly grid."""
    timestamps = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_eur_mwh": np.linspace(40.0, 60.0, n),
        }
    )
    if include_load:
        frame["load_mw"] = np.linspace(9_000.0, 11_000.0, n)
    return frame


def _generation_frame(n=96):
    """What the (mocked) generation download returns, same hourly grid."""
    timestamps = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "wind_mw": np.linspace(500.0, 2_500.0, n),
            "solar_mw": np.linspace(0.0, 1_500.0, n),
        }
    )


def _run_ingest(cfg, *, natural=False):
    """Run ``ingestion.main.main()``; return the raw CSV path.

    The ENTSOE_API_KEY is blanked so a key in the developer environment
    can never trigger a live call. With ``natural=False`` the price and
    generation downloads are mocked (manifest source ``entsoe``); with
    ``natural=True`` nothing is mocked and the pipeline exercises the
    real all-synthetic offline fallback (source ``synthetic`` — decision
    D7). Weather is disabled in every config here, so no network I/O is
    possible either way.
    """
    with (
        mock.patch("src.ingestion.main.load_config", return_value=cfg),
        mock.patch.dict(os.environ, {"ENTSOE_API_KEY": ""}),
    ):
        if natural:
            ingest_main()
        else:
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "src.ingestion.main.download_entsoe_data",
                        return_value=_download_frame(
                            cfg.data.entsoe.include_load
                        ),
                    )
                )
                if cfg.features.generation_mix.enabled:
                    stack.enter_context(
                        mock.patch(
                            "src.ingestion.entsoe.download_generation_mix",
                            return_value=_generation_frame(),
                        )
                    )
                ingest_main()
    return Path(cfg.data.raw_path) / "entsoe_prices.csv"


def _manifest_of(cfg):
    """Load ``manifest.json`` from the config's raw directory."""
    path = Path(cfg.data.raw_path) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_features(tmp_path, cfg, raw, weather=None):
    """build_features → split → save parquet + schema sidecar.

    Returns ``(features_parquet_path, schema_path, schema)``.
    """
    out, schema = build_features(
        raw, cfg.features, cfg.data.target_col, weather=weather
    )
    train, val, test = train_val_test_split(out, cfg.data)
    processed = tmp_path / "processed" / "features.parquet"
    save_processed_data(
        train,
        val,
        test,
        str(processed),
        str(tmp_path / "reference" / "reference.parquet"),
    )
    schema_path = processed.with_name("features_schema.json")
    schema.save(schema_path)
    return str(processed), schema_path, schema


# ---------------------------------------------------------------------------
# 1. Ingest output — data/raw/entsoe_prices.csv + manifest.json
# ---------------------------------------------------------------------------


class TestIngestOutputContract:
    """Pin the 'ingest output' row of docs/contracts.md §3."""

    def test_default_output_exact_columns(self, tmp_path):
        """Positive: exact column set for the default flag combination."""
        cfg = _make_config(raw_path=tmp_path / "raw")
        df = pd.read_csv(_run_ingest(cfg), parse_dates=["timestamp"])
        assert set(df.columns) == {
            "timestamp",
            "price_eur_mwh",
            "load_mw",
            "is_imputed",
        }
        assert len(df) > 0

    def test_timestamp_is_tz_aware_utc(self, tmp_path):
        """Positive: timestamp parses as tz-aware UTC (§3 required dtype)."""
        cfg = _make_config(raw_path=tmp_path / "raw")
        df = pd.read_csv(_run_ingest(cfg), parse_dates=["timestamp"])
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_price_eur_mwh_is_float(self, tmp_path):
        """Positive: the target column is float (§3 required dtype)."""
        cfg = _make_config(raw_path=tmp_path / "raw")
        df = pd.read_csv(_run_ingest(cfg))
        assert pd.api.types.is_float_dtype(df["price_eur_mwh"])

    def test_is_imputed_present_when_flag_enabled(self, tmp_path):
        """Positive: add_is_imputed_flag=True → boolean column present."""
        cfg = _make_config(raw_path=tmp_path / "raw", add_is_imputed=True)
        df = pd.read_csv(_run_ingest(cfg))
        assert df["is_imputed"].dtype == bool

    def test_is_imputed_absent_when_flag_disabled(self, tmp_path):
        """Negative: add_is_imputed_flag=False → no is_imputed column."""
        cfg = _make_config(raw_path=tmp_path / "raw", add_is_imputed=False)
        df = pd.read_csv(_run_ingest(cfg))
        assert "is_imputed" not in df.columns
        assert set(df.columns) == {"timestamp", "price_eur_mwh", "load_mw"}

    def test_load_mw_present_when_include_load(self, tmp_path):
        """Positive: include_load=True → float load_mw column present."""
        cfg = _make_config(raw_path=tmp_path / "raw", include_load=True)
        df = pd.read_csv(_run_ingest(cfg))
        assert "load_mw" in df.columns
        assert pd.api.types.is_float_dtype(df["load_mw"])

    def test_load_mw_absent_when_include_load_false(self, tmp_path):
        """Negative: include_load=False → no load_mw column."""
        cfg = _make_config(raw_path=tmp_path / "raw", include_load=False)
        df = pd.read_csv(_run_ingest(cfg))
        assert "load_mw" not in df.columns
        assert set(df.columns) == {"timestamp", "price_eur_mwh", "is_imputed"}

    def test_generation_mw_columns_present_when_enabled(self, tmp_path):
        """Positive: generation flag on → every {source}_mw persisted."""
        cfg = _make_config(raw_path=tmp_path / "raw", **_generation_flags())
        df = pd.read_csv(_run_ingest(cfg), parse_dates=["timestamp"])
        assert set(df.columns) == {
            "timestamp",
            "price_eur_mwh",
            "load_mw",
            "is_imputed",
            "wind_mw",
            "solar_mw",
        }
        assert pd.api.types.is_float_dtype(df["wind_mw"])
        assert pd.api.types.is_float_dtype(df["solar_mw"])
        # No duplicate-merge corruption (wind_mw_x / wind_mw_y).
        assert not any(c.endswith(("_x", "_y")) for c in df.columns)

    def test_generation_mw_columns_absent_when_disabled(self, tmp_path):
        """Negative: generation flag off → no generation columns."""
        cfg = _make_config(raw_path=tmp_path / "raw")
        df = pd.read_csv(_run_ingest(cfg))
        assert set(df.columns) == {
            "timestamp",
            "price_eur_mwh",
            "load_mw",
            "is_imputed",
        }

    def test_no_weather_columns_in_raw_csv(self, tmp_path):
        """Weather is cached separately — never in entsoe_prices.csv (§3)."""
        cfg = _make_config(raw_path=tmp_path / "raw")
        df = pd.read_csv(_run_ingest(cfg))
        assert not any("__" in c for c in df.columns)

    def test_manifest_has_required_fields(self, tmp_path):
        """Positive: manifest carries provenance + imputation stats."""
        cfg = _make_config(raw_path=tmp_path / "raw")
        _run_ingest(cfg)
        manifest = _manifest_of(cfg)
        for key in (
            "downloaded_at",
            "date_range",
            "row_count",
            "sha256",
            "source",
            "n_imputed_rows",
            "n_dropped_rows",
            "n_unfilled_rows",
            "max_gap_periods",
            "fill_method",
            "freq",
        ):
            assert key in manifest, f"manifest missing {key!r}"
        assert manifest["source"] in {"entsoe", "synthetic"}

    def test_manifest_sha256_and_row_count_match_csv(self, tmp_path):
        """Positive: manifest hash/rows describe the on-disk CSV exactly."""
        cfg = _make_config(raw_path=tmp_path / "raw")
        csv_path = _run_ingest(cfg)
        manifest = _manifest_of(cfg)
        on_disk = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        assert manifest["sha256"] == on_disk
        assert manifest["row_count"] == len(pd.read_csv(csv_path))
        assert manifest["row_count"] > 0

    def test_missing_api_key_without_opt_in_raises(self, tmp_path):
        """Option A: no key + no explicit opt-in → loud, actionable error.

        Production can never silently fall back to fabricated data —
        aligned with ``features.weather.allow_synthetic`` (decision D7).
        """
        cfg = _make_config(raw_path=tmp_path / "raw")
        with pytest.raises(RuntimeError, match="allow_synthetic"):
            _run_ingest(cfg, natural=True)

    def test_natural_synthetic_fallback_offline(self, tmp_path):
        """Explicit opt-in: the all-synthetic path (decision D7).

        Exercises the real fallback wiring end-to-end — price, load, and
        generation all synthetic and source-consistent — and pins that
        the generation columns persist with no duplicate merge suffixes.
        ``main()`` raising here would mean the source-consistency gate
        rejected a mixed run.
        """
        cfg = _make_config(
            raw_path=tmp_path / "raw", allow_synthetic=True, **_generation_flags()
        )
        csv_path = _run_ingest(cfg, natural=True)
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        assert set(df.columns) == {
            "timestamp",
            "price_eur_mwh",
            "load_mw",
            "is_imputed",
            "wind_mw",
            "solar_mw",
        }
        assert not any(c.endswith(("_x", "_y")) for c in df.columns)
        assert len(df) > 0
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert _manifest_of(cfg)["source"] == "synthetic"


# ---------------------------------------------------------------------------
# 2. Featurise output — features DataFrame + FeatureSchema sidecar
# ---------------------------------------------------------------------------


class TestFeaturiseOutputContract:
    """Pin the 'featurise output' row of docs/contracts.md §3."""

    def test_all_feature_groups_disabled_exact_columns(self):
        """Positive: every group off → the ingest columns pass through."""
        raw = _raw_df()
        cfg = _make_config()
        out, schema = build_features(raw, cfg.features, cfg.data.target_col)
        assert set(out.columns) == set(raw.columns)
        schema.assert_matches_dataframe(out)

    def test_calendar_columns_when_enabled(self):
        """Positive: calendar flag on → exactly the configured names."""
        raw = _raw_df()
        cfg = _make_config(**_calendar_flags())
        out, _ = build_features(raw, cfg.features, cfg.data.target_col)
        assert set(out.columns) == set(raw.columns) | set(CALENDAR_COLUMNS)

    def test_lag_columns_match_config_periods(self):
        """Positive: lag_{p}h for each configured period, no others."""
        raw = _raw_df()
        cfg = _make_config(**_lags_flags())
        out, _ = build_features(raw, cfg.features, cfg.data.target_col)
        lag_cols = {c for c in out.columns if c.startswith("lag_")}
        assert lag_cols == {"lag_1h", "lag_24h"}

    def test_rolling_columns_match_config_windows(self):
        """Positive: rolling_{stat}_{w}h for each configured window."""
        raw = _raw_df()
        cfg = _make_config(**_lags_flags())
        out, _ = build_features(raw, cfg.features, cfg.data.target_col)
        rolling_cols = {c for c in out.columns if c.startswith("rolling_")}
        assert rolling_cols == {"rolling_mean_24h", "rolling_std_24h"}

    def test_derivative_columns_when_enabled(self):
        """Positive: {target}_diff_{n} for each configured order."""
        raw = _raw_df()
        cfg = _make_config(
            derivatives={"enabled": True, "order": [1, 2], "smooth_window": 3}
        )
        out, _ = build_features(raw, cfg.features, cfg.data.target_col)
        assert set(out.columns) - set(raw.columns) == {
            "price_eur_mwh_diff_1",
            "price_eur_mwh_diff_2",
        }

    def test_weather_columns_when_enabled(self):
        """Positive: {location}__{variable} names (decision D9), pinned."""
        raw = _raw_df()
        cfg = _make_config(**_weather_flags())
        out, schema = build_features(
            raw, cfg.features, cfg.data.target_col, weather=_weather_frames(raw)
        )
        weather_cols = {c for c in out.columns if "__" in c}
        assert weather_cols == {
            "warsaw__temperature_2m",
            "warsaw__wind_speed_100m",
        }
        assert set(schema.select(group="weather")) == weather_cols

    def test_no_weather_columns_when_disabled(self):
        """Negative: weather flag off → no {location}__{variable} columns."""
        raw = _raw_df()
        cfg = _make_config(**_calendar_flags())
        out, _ = build_features(raw, cfg.features, cfg.data.target_col)
        assert not any("__" in c for c in out.columns)

    def test_timestamp_stays_tz_aware_after_weather_merge(self):
        """Regression: the weather merge must not strip the timezone.

        ``merge_weather`` normalized to tz-naive UTC, which broke
        ``get_split_masks`` downstream (it compares against tz-aware
        boundaries) and contradicted the schema's declared dtype.
        """
        raw = _raw_df()
        cfg = _make_config(**_weather_flags())
        out, schema = build_features(
            raw, cfg.features, cfg.data.target_col, weather=_weather_frames(raw)
        )
        assert str(out["timestamp"].dt.tz) == "UTC"
        assert schema.columns["timestamp"].dtype == str(out["timestamp"].dtype)

    def test_availability_lags_replace_raw_externals(self):
        """No-leakage: raw externals never survive when lagged (WS4)."""
        raw = _raw_df(include_generation=True)
        cfg = _make_config(**_generation_flags())
        out, schema = build_features(raw, cfg.features, cfg.data.target_col)
        assert not {"load_mw", "wind_mw", "solar_mw"} & set(out.columns)
        lagged = {"load_mw_lag1h", "wind_mw_lag1h", "solar_mw_lag1h"}
        assert lagged <= set(out.columns)
        assert set(schema.select(group="availability_lag")) == lagged

    def test_raw_external_kept_when_no_lag_configured(self):
        """Positive: no availability_lags entry → the raw column stays."""
        raw = _raw_df()
        cfg = _make_config()
        out, schema = build_features(raw, cfg.features, cfg.data.target_col)
        assert "load_mw" in out.columns
        assert schema.columns["load_mw"].group == "external"

    def test_schema_drift_guard_and_order(self):
        """Positive: schema covers exactly the frame, in frame order."""
        raw = _raw_df(include_generation=True)
        cfg = _make_config(
            **_calendar_flags(),
            **_lags_flags(),
            **_generation_flags(),
            **_weather_flags(),
            derivatives={"enabled": True, "order": [1, 2], "smooth_window": 3},
        )
        out, schema = build_features(
            raw, cfg.features, cfg.data.target_col, weather=_weather_frames(raw)
        )
        schema.assert_matches_dataframe(out)
        assert schema.names() == list(out.columns)

    def test_feature_names_exclude_identifier_target_meta(self):
        """Positive: feature_names() is exactly the model matrix X."""
        raw = _raw_df(include_generation=True)
        cfg = _make_config(**_calendar_flags(), **_generation_flags())
        out, schema = build_features(raw, cfg.features, cfg.data.target_col)
        assert schema.columns["timestamp"].role == ColumnRole.IDENTIFIER
        assert schema.columns["price_eur_mwh"].role == ColumnRole.TARGET
        assert schema.columns["is_imputed"].role == ColumnRole.META
        names = schema.feature_names()
        assert "timestamp" not in names
        assert "price_eur_mwh" not in names
        assert "is_imputed" not in names
        assert "hour" in names
        assert names == [c for c in out.columns if c in names]

    def test_full_config_exact_column_set_and_dtypes(self):
        """The §3 contract end-to-end: every group on, exact output."""
        raw = _raw_df(include_generation=True)
        cfg = _make_config(
            **_calendar_flags(),
            **_lags_flags(),
            **_generation_flags(),
            **_weather_flags(),
            derivatives={"enabled": True, "order": [1, 2], "smooth_window": 3},
        )
        out, schema = build_features(
            raw, cfg.features, cfg.data.target_col, weather=_weather_frames(raw)
        )
        expected = (
            {"timestamp", "price_eur_mwh", "is_imputed"}
            | set(CALENDAR_COLUMNS)
            | {"load_mw_lag1h", "wind_mw_lag1h", "solar_mw_lag1h"}
            | {"warsaw__temperature_2m", "warsaw__wind_speed_100m"}
            | {"lag_1h", "lag_24h", "rolling_mean_24h", "rolling_std_24h"}
            | {"price_eur_mwh_diff_1", "price_eur_mwh_diff_2"}
        )
        assert set(out.columns) == expected
        # The final dropna gate leaves no NaN anywhere (§3 / decision D6).
        assert not out.isna().any().any()
        # dtypes per the contract:
        assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"
        assert out["price_eur_mwh"].dtype == np.float64
        assert out["is_imputed"].dtype == bool
        assert str(out["days_to_next_holiday"].dtype) == "Int64"
        schema.assert_matches_dataframe(out)


# ---------------------------------------------------------------------------
# 3. Train input — load_features arrays vs get_feature_names / split masks
# ---------------------------------------------------------------------------


class TestTrainInputContract:
    """Pin the 'train/evaluate input' row of docs/contracts.md §3."""

    @pytest.fixture
    def feature_bundle(self, tmp_path):
        """features.parquet + schema sidecar from a contract-shaped frame."""
        raw = _raw_df()
        cfg = _make_config(**_calendar_flags(), **_lags_flags())
        path, schema_path, schema = _write_features(tmp_path, cfg, raw)
        return path, schema_path, schema, cfg

    def test_feature_count_matches_get_feature_names(self, feature_bundle):
        """Positive: X column count equals len(get_feature_names(...))."""
        path, _, _, cfg = feature_bundle
        names = get_feature_names(path, cfg.data)
        X_train, _, X_val, _, X_test, _ = load_features(path, cfg.data)
        assert len(names) > 0
        assert X_train.shape[1] == len(names)
        assert X_val.shape[1] == len(names)
        assert X_test.shape[1] == len(names)

    def test_row_counts_match_split_masks(self, feature_bundle):
        """Positive: split array lengths equal the split masks exactly."""
        path, _, _, cfg = feature_bundle
        df = pd.read_parquet(path)
        train_mask, val_mask, test_mask = get_split_masks(df, cfg.data)
        X_train, y_train, X_val, y_val, X_test, y_test = load_features(
            path, cfg.data
        )
        assert X_train.shape[0] == int(train_mask.sum()) == y_train.shape[0]
        assert X_val.shape[0] == int(val_mask.sum()) == y_val.shape[0]
        assert X_test.shape[0] == int(test_mask.sum()) == y_test.shape[0]
        total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        assert total == len(df)

    def test_excludes_identifier_and_target(self, feature_bundle):
        """Positive: X never contains timestamp or the target column."""
        path, _, _, cfg = feature_bundle
        names = get_feature_names(path, cfg.data)
        assert "timestamp" not in names
        assert "price_eur_mwh" not in names
        assert any(n.startswith("lag_") for n in names)

    def test_schema_sidecar_drives_selection(self, feature_bundle):
        """Positive: with a sidecar, names come from role==FEATURE."""
        path, _, schema, cfg = feature_bundle
        names = get_feature_names(path, cfg.data)
        assert names == schema.feature_names()
        # The sidecar's FEATURE role excludes the meta column...
        assert "is_imputed" not in names

    def test_legacy_fallback_without_sidecar(self, feature_bundle, caplog):
        """Negative: missing sidecar → legacy rule + a WARNING, no crash."""
        path, schema_path, _, cfg = feature_bundle
        schema_path.unlink()
        with caplog.at_level(logging.WARNING, logger="src.training.loader"):
            names = get_feature_names(path, cfg.data)
        # Legacy rule: every column except timestamp and the target —
        # including meta columns the sidecar would have excluded.
        df = pd.read_parquet(path)
        assert set(names) == set(df.columns) - {"timestamp", "price_eur_mwh"}
        assert "is_imputed" in names
        assert "No feature schema sidecar" in caplog.text
