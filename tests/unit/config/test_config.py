"""
Unit tests for the config package — PipelineConfig schema validation
and the params.yaml loader.

Validation must fail at load time, before any pipeline stage runs.
"""

from datetime import date
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError

from src.common.metrics import compute_metrics
from src.config import load_config
from src.config.models import (
    SUPPORTED_METRICS,
    CalendarFeature,
    CrossBorderFlowsConfig,
    DataConfig,
    EntsoeConfig,
    FeatureSelection,
    LoggingConfig,
    ModelConfig,
    PipelineConfig,
    TemporalConfig,
)


def _minimal_raw() -> dict:
    """A minimal raw config dict that validates via model defaults."""
    return {
        "data": {
            "target_col": "price_eur_mwh",
            "train_end": "2023-12-31",
            "val_end": "2024-01-01",
        },
        "model": {"type": "sklearn.dummy.DummyRegressor"},
    }


# ---------------------------------------------------------------------------
# load_config (loader)
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_loads_real_params_yaml(self):
        """Positive: the project's params.yaml validates against the schema."""
        cfg = load_config("params.yaml")
        assert isinstance(cfg, PipelineConfig)
        assert cfg.data.target_col == "price_eur_mwh"
        assert cfg.data.train_end == date(2023, 12, 31)
        assert cfg.data.val_end == date(2024, 6, 30)
        assert cfg.model.type == "xgboost.XGBRegressor"
        # Stage 4 step 1 (§7.1): additive config keys
        assert cfg.data.superset_dir == "data/processed/superset/"
        assert cfg.temporal.strategy == "single_step"
        assert cfg.features.select is None
        assert cfg.features.cross_border_flows.enabled is False
        assert set(cfg.features.cross_border_flows.neighbours) == {
            "DE_LU",
            "CZ",
            "SK",
            "LT",
            "SE_4",
        }
        assert cfg.features.weather.derivatives.enabled is False
        assert "is_long_weekend" in cfg.features.calendar.include
        assert cfg.evaluation.sample_week == date(2024, 3, 4)
        assert cfg.evaluation.naive_period == 168
        assert cfg.evaluation.dm_test is True

    def test_minimal_yaml_fills_defaults(self, tmp_path):
        """Positive: a minimal config validates; defaults fill the rest."""
        cfg_path = tmp_path / "params.yaml"
        cfg_path.write_text(
            "data:\n"
            "  target_col: price_eur_mwh\n"
            "  train_end: '2023-12-31'\n"
            "  val_end: '2024-01-01'\n"
            "model:\n"
            "  type: sklearn.dummy.DummyRegressor\n"
        )
        cfg = load_config(str(cfg_path))
        assert cfg.temporal.resolution == "hourly"
        assert cfg.mlflow.champion_alias == "champion"
        assert cfg.logging.level == "INFO"

    def test_missing_file_raises(self, tmp_path):
        """Negative: missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "does_not_exist.yaml"))

    def test_invalid_yaml_raises(self, tmp_path):
        """Negative: malformed YAML raises yaml.YAMLError."""
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("data: [unclosed")
        with pytest.raises(yaml.YAMLError):
            load_config(str(cfg_path))


# ---------------------------------------------------------------------------
# PipelineConfig schema
# ---------------------------------------------------------------------------
class TestPipelineConfigSchema:
    def test_minimal_raw_dict_validates(self):
        """Positive: minimal dict validates; defaults fill every other key."""
        cfg = PipelineConfig.model_validate(_minimal_raw())
        assert cfg.data.target_col == "price_eur_mwh"
        assert cfg.data.train_end == date(2023, 12, 31)
        assert cfg.model.type == "sklearn.dummy.DummyRegressor"
        assert cfg.features.lags.enabled is False

    def test_missing_required_key_raises(self):
        """Negative: missing required key fails before any stage runs."""
        raw = _minimal_raw()
        del raw["model"]["type"]
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_missing_required_section_raises(self):
        """Negative: missing required section fails validation."""
        raw = _minimal_raw()
        del raw["data"]
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_wrong_type_raises(self):
        """Negative: a list where an int is expected fails validation."""
        raw = _minimal_raw()
        raw["temporal"] = {"horizon": [24]}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_unknown_key_raises(self):
        """Negative: a typo'd key (extra='forbid') fails validation."""
        raw = _minimal_raw()
        raw["data"]["raw_pat"] = "data/raw/"
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_train_end_after_val_end_raises(self):
        """Negative: inverted split boundaries fail validation."""
        raw = _minimal_raw()
        raw["data"]["train_end"] = "2024-06-30"
        raw["data"]["val_end"] = "2023-12-31"
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_train_end_equal_val_end_raises(self):
        """Negative: empty val window (train_end == val_end) fails validation."""
        raw = _minimal_raw()
        raw["data"]["val_end"] = raw["data"]["train_end"]
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_unknown_metric_raises(self):
        """Negative: metric names outside the supported set fail validation."""
        raw = _minimal_raw()
        raw["evaluation"] = {"metrics": ["rmse", "smape"]}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_primary_metric_not_in_metrics_raises(self):
        """Negative: primary_metric must be among the computed metrics."""
        raw = _minimal_raw()
        raw["evaluation"] = {"primary_metric": "mae", "metrics": ["rmse"]}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_model_type_without_dot_raises(self):
        """Negative: model.type must be a dotted import path."""
        raw = _minimal_raw()
        raw["model"] = {"type": "XGBRegressor"}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_model_type_not_identifier_raises(self):
        """Negative: model.type with invalid path segments fails validation."""
        raw = _minimal_raw()
        raw["model"] = {"type": "xgboost.XGB Regressor"}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_invalid_resolution_raises(self):
        """Negative: resolution outside {hourly, daily, weekly} fails."""
        raw = _minimal_raw()
        raw["temporal"] = {"resolution": "minutely"}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_invalid_fill_method_raises(self):
        """Negative: fill_method outside {ffill, interpolate} fails."""
        raw = _minimal_raw()
        raw["data"]["fill_method"] = "bfill"
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_non_positive_lag_period_raises(self):
        """Negative: lag periods must be positive ints."""
        raw = _minimal_raw()
        raw["features"] = {"lags": {"enabled": True, "periods": [0, 24]}}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_invalid_weather_location_raises(self):
        """Negative: empty or uppercase location names fail validation."""
        raw = _minimal_raw()
        raw["features"] = {"weather": {"enabled": True, "locations": ["Warsaw"]}}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_logging_level_normalized_to_upper(self):
        """Positive: lowercase logging level is accepted and normalized."""
        raw = _minimal_raw()
        raw["logging"] = {"level": "debug"}
        cfg = PipelineConfig.model_validate(raw)
        assert cfg.logging.level == "DEBUG"


# ---------------------------------------------------------------------------
# LagsConfig (Workstream 2)
# ---------------------------------------------------------------------------
class TestLagsConfig:
    def test_rolling_windows_default(self):
        """Positive: rolling windows default to [24, 168] when omitted."""
        cfg = PipelineConfig.model_validate(_minimal_raw())
        assert cfg.features.lags.rolling_windows == [24, 168]

    def test_periods_deduped_and_sorted(self):
        """Positive: duplicate/unsorted periods normalize to unique sorted."""
        raw = _minimal_raw()
        raw["features"] = {"lags": {"enabled": True, "periods": [168, 24, 24, 1]}}
        cfg = PipelineConfig.model_validate(raw)
        assert cfg.features.lags.periods == [1, 24, 168]

    def test_non_positive_rolling_window_raises(self):
        """Negative: rolling windows must be positive ints."""
        raw = _minimal_raw()
        raw["features"] = {"lags": {"enabled": True, "rolling_windows": [0, 24]}}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Stage 4 step 1 (§7.1) — additive config keys (WS0/WS1/WS2)
# ---------------------------------------------------------------------------
class TestFeatureSelection:
    """``features.select`` — train-time column subset (WS0)."""

    def test_groups_only_validates(self):
        """Positive: a groups-only selection validates."""
        sel = FeatureSelection(groups=["calendar", "lags"])
        assert sel.groups == ["calendar", "lags"]
        assert sel.columns == []

    def test_columns_only_validates(self):
        """Positive: an explicit column list validates."""
        sel = FeatureSelection(columns=["hour", "price_diff_1"])
        assert sel.columns == ["hour", "price_diff_1"]
        assert sel.groups == []

    def test_groups_and_columns_combine(self):
        """Positive: groups (union) and explicit columns may be combined."""
        sel = FeatureSelection(groups=["calendar"], columns=["load_mw_lag1h"])
        assert sel.groups == ["calendar"]
        assert sel.columns == ["load_mw_lag1h"]

    def test_empty_selection_raises(self):
        """Negative: an explicit selection must pick at least one thing."""
        with pytest.raises(ValidationError, match="features.select"):
            FeatureSelection(groups=[], columns=[])

    @pytest.mark.parametrize(
        "raw",
        [
            {"groups": [""]},
            {"columns": ["   "]},
            {"groups": ["calendar"], "columns": [""]},
        ],
    )
    def test_blank_entry_raises(self, raw):
        """Negative: blank group/column names are typos, not selections."""
        with pytest.raises(ValidationError, match="features.select"):
            FeatureSelection(**raw)

    def test_default_select_is_none(self):
        """Positive: features.select defaults to None = all superset columns."""
        cfg = PipelineConfig.model_validate(_minimal_raw())
        assert cfg.features.select is None

    def test_select_inside_pipeline_config(self):
        """Positive: features.select parses as a nested model."""
        raw = _minimal_raw()
        raw["features"] = {"select": {"groups": ["calendar"]}}
        cfg = PipelineConfig.model_validate(raw)
        assert cfg.features.select is not None
        assert cfg.features.select.groups == ["calendar"]


class TestDataConfigSupersetDir:
    """``data.superset_dir`` — where materialised supersets live (WS0)."""

    def test_default_superset_dir(self):
        """Positive: defaults to the WS0 materialisation path."""
        data = DataConfig.model_validate(
            {
                "target_col": "price_eur_mwh",
                "train_end": "2023-12-31",
                "val_end": "2024-01-01",
            }
        )
        assert data.superset_dir == "data/processed/superset/"

    def test_superset_dir_override(self):
        """Positive: an explicit path is kept as written."""
        data = DataConfig.model_validate(
            {
                "target_col": "price_eur_mwh",
                "train_end": "2023-12-31",
                "val_end": "2024-01-01",
                "superset_dir": "custom/superset/",
            }
        )
        assert data.superset_dir == "custom/superset/"


class TestTemporalStrategy:
    """``temporal.strategy`` — horizon-training strategy (WS1, EXP-014/015)."""

    def test_default_is_single_step(self):
        """Positive: the default preserves the Stage-3 training behaviour."""
        assert TemporalConfig().strategy == "single_step"

    @pytest.mark.parametrize("strategy", ["direct", "shared"])
    def test_ws1_strategies_accepted(self, strategy):
        """Positive: the WS1 horizon strategies validate."""
        assert TemporalConfig(strategy=strategy).strategy == strategy

    def test_unknown_strategy_raises(self):
        """Negative: a typo in temporal.strategy fails at load."""
        with pytest.raises(ValidationError):
            TemporalConfig(strategy="one_step")


class TestCrossBorderFlowsConfig:
    """``features.cross_border_flows.neighbours`` (WS2, EXP-006)."""

    def test_defaults(self):
        """Positive: disabled by default with the plan's ENTSO-E zones."""
        cfg = CrossBorderFlowsConfig()
        assert cfg.enabled is False
        assert set(cfg.neighbours) == {"DE_LU", "CZ", "SK", "LT", "SE_4"}

    def test_neighbours_deduped_and_sorted(self):
        """Positive: duplicates collapse and order normalises (mirror of sources)."""
        cfg = CrossBorderFlowsConfig(neighbours=["SK", "CZ", "SK"])
        assert cfg.neighbours == ["CZ", "SK"]

    def test_unknown_zone_raises(self):
        """Negative: an ENTSO-E area typo fails at load."""
        with pytest.raises(ValidationError):
            CrossBorderFlowsConfig(neighbours=["DE_LU", "XX"])


# ---------------------------------------------------------------------------
# availability_lags × cross-border flows (WS2, EXP-006)
# ---------------------------------------------------------------------------
class TestAvailabilityLagsFlows:
    """Flow columns obey the same availability-lag contract as generation."""

    def test_flow_key_accepted(self):
        """Positive: a known flow column may declare its publication lag."""
        raw = _minimal_raw()
        raw["features"] = {"availability_lags": {"flow_de_lu_mw": 1}}
        cfg = PipelineConfig.model_validate(raw)
        assert cfg.features.availability_lags["flow_de_lu_mw"] == 1

    def test_unknown_flow_zone_raises(self):
        """Negative: a flow-zone typo is still rejected at load."""
        raw = _minimal_raw()
        raw["features"] = {"availability_lags": {"flow_xx_mw": 1}}
        with pytest.raises(ValidationError, match="availability_lags"):
            PipelineConfig.model_validate(raw)

    def test_enabled_flows_require_lag_entries(self):
        """Negative: enabled flows without lag entries fail at load."""
        raw = _minimal_raw()
        raw["features"] = {"cross_border_flows": {"enabled": True}}
        with pytest.raises(ValidationError, match="cross_border_flows"):
            PipelineConfig.model_validate(raw)

    def test_enabled_flows_with_entries_validate(self):
        """Positive: one lag entry per enabled neighbour is accepted."""
        raw = _minimal_raw()
        raw["features"] = {
            "cross_border_flows": {"enabled": True, "neighbours": ["CZ"]},
            "availability_lags": {"flow_cz_mw": 1},
        }
        cfg = PipelineConfig.model_validate(raw)
        assert cfg.features.cross_border_flows.enabled is True
        assert cfg.features.availability_lags["flow_cz_mw"] == 1

    def test_disabled_flows_impose_no_requirement(self):
        """Positive: flows off (default) need no availability_lags entries."""
        cfg = PipelineConfig.model_validate(_minimal_raw())
        assert cfg.features.cross_border_flows.enabled is False
        assert cfg.features.availability_lags == {}


# ---------------------------------------------------------------------------
# features.weather.derivatives (WS3, EXP-011)
# ---------------------------------------------------------------------------
class TestWeatherDerivatives:
    """``features.weather.derivatives`` — weather-derivative controls."""

    def test_defaults_are_inert(self):
        """Positive: disabled by default with the plan's default block."""
        cfg = PipelineConfig.model_validate(_minimal_raw())
        wd = cfg.features.weather.derivatives
        assert wd.enabled is False
        assert wd.variables == ["wind_speed_100m"]
        assert wd.order == [1]
        assert wd.smooth_windows == [3]

    def test_enabled_subset_validates(self):
        """Positive: derivative variables drawn from weather variables pass."""
        raw = _minimal_raw()
        raw["features"] = {
            "weather": {
                "enabled": True,
                "variables": ["temperature_2m", "wind_speed_100m"],
                "derivatives": {"enabled": True, "variables": ["wind_speed_100m"]},
            }
        }
        cfg = PipelineConfig.model_validate(raw)
        assert cfg.features.weather.derivatives.enabled is True

    def test_variable_outside_weather_variables_raises(self):
        """Negative: a derivative over an unconfigured variable fails at load."""
        raw = _minimal_raw()
        raw["features"] = {
            "weather": {
                "variables": ["temperature_2m"],
                "derivatives": {"enabled": True, "variables": ["wind_speed_100m"]},
            }
        }
        with pytest.raises(ValidationError, match="derivatives"):
            PipelineConfig.model_validate(raw)

    def test_disabled_block_ignores_subset(self):
        """Positive: derivatives off ⇒ narrowed weather.variables still loads."""
        raw = _minimal_raw()
        raw["features"] = {
            "weather": {
                "variables": ["temperature_2m"],
                "derivatives": {"enabled": False, "variables": ["wind_speed_100m"]},
            }
        }
        cfg = PipelineConfig.model_validate(raw)
        assert cfg.features.weather.derivatives.enabled is False

    @pytest.mark.parametrize("field", [{"smooth_window": 0}, {"order": [0]}])
    def test_non_positive_values_raise(self, field):
        """Negative: smooth_window and order must be positive ints."""
        raw = _minimal_raw()
        raw["features"] = {"weather": {"derivatives": field}}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# calendar.include + is_long_weekend (WS3, EXP-007)
# ---------------------------------------------------------------------------
class TestCalendarInclude:
    """``CalendarFeature`` vocabulary and the config-driven include gate."""

    def test_is_long_weekend_in_vocabulary(self):
        """Contract: the CalendarFeature Literal admits is_long_weekend."""
        assert "is_long_weekend" in get_args(CalendarFeature)

    def test_default_include_contains_is_long_weekend(self):
        """Positive: the default include list ships the new feature."""
        cfg = PipelineConfig.model_validate(_minimal_raw())
        assert "is_long_weekend" in cfg.features.calendar.include

    def test_unknown_calendar_feature_raises(self):
        """Negative: a typo in include still fails at load."""
        raw = _minimal_raw()
        raw["features"] = {"calendar": {"include": ["is_long_weekendx"]}}
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# evaluation — WS4 residual controls + WS7 metric plumbing keys
# ---------------------------------------------------------------------------
class TestEvaluationControls:
    """Additive ``evaluation`` keys: residual analysis (WS4), naive/DM (WS7)."""

    def test_defaults_preserve_behaviour(self):
        """Positive: new keys default to inert, plan-specified values."""
        cfg = PipelineConfig.model_validate(_minimal_raw())
        ev = cfg.evaluation
        assert ev.residual_quantiles == 5
        assert ev.sample_week is None
        assert ev.run_analysis_after_train is True
        assert ev.naive_period == 168
        assert ev.dm_test is True

    @pytest.mark.parametrize("bad", [-1, -5])
    def test_negative_quantiles_raise(self, bad):
        """Negative: negative bin counts are nonsense (0 disables)."""
        raw = _minimal_raw()
        raw["evaluation"] = {"residual_quantiles": bad}
        with pytest.raises(ValidationError, match="residual_quantiles"):
            PipelineConfig.model_validate(raw)

    def test_zero_quantiles_disables(self):
        """Positive: 0 disables the price-quantile breakdown."""
        raw = _minimal_raw()
        raw["evaluation"] = {"residual_quantiles": 0}
        cfg = PipelineConfig.model_validate(raw)
        assert cfg.evaluation.residual_quantiles == 0

    @pytest.mark.parametrize("field", [{"naive_period": 0}, {"naive_period": -168}])
    def test_non_positive_naive_period_raises(self, field):
        """Negative: naive_period is a PositiveInt count of native steps."""
        raw = _minimal_raw()
        raw["evaluation"] = field
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate(raw)

    def test_sample_week_parses_to_date(self):
        """Positive: the YAML ISO week-start string parses to datetime.date."""
        raw = _minimal_raw()
        raw["evaluation"] = {"sample_week": "2024-03-04"}
        cfg = PipelineConfig.model_validate(raw)
        assert cfg.evaluation.sample_week == date(2024, 3, 4)


# ---------------------------------------------------------------------------
# Section models used directly by consumers
# ---------------------------------------------------------------------------
class TestSectionModels:
    def test_data_config_split_dates_parsed_to_date(self):
        """Positive: YAML date strings are parsed to datetime.date."""
        data = DataConfig.model_validate(
            {
                "target_col": "price_eur_mwh",
                "train_end": "2023-12-31",
                "val_end": "2024-01-01",
            }
        )
        assert data.train_end == date(2023, 12, 31)
        assert isinstance(data.train_end, date)

    def test_temporal_pandas_freq_mapping(self):
        """Positive: each resolution maps to its pandas frequency alias."""
        assert TemporalConfig(resolution="hourly").pandas_freq == "h"
        assert TemporalConfig(resolution="daily").pandas_freq == "D"
        assert TemporalConfig(resolution="weekly").pandas_freq == "W"

    def test_model_config_dotted_path(self):
        """Positive: dotted model type is accepted and params kept."""
        model = ModelConfig(type="xgboost.XGBRegressor", params={"n_estimators": 5})
        assert model.type == "xgboost.XGBRegressor"
        assert model.params == {"n_estimators": 5}

    def test_logging_config_default_file_is_none(self):
        """Positive: LoggingConfig.file defaults to None (DEFAULT_LOG_FILE)."""
        assert LoggingConfig().file is None
        assert LoggingConfig().level == "INFO"


# ---------------------------------------------------------------------------
# EntsoeConfig bidding-zone normalization
# ---------------------------------------------------------------------------
class TestEntsoeConfigZone:
    def test_zone_normalized(self):
        """Positive: whitespace/case normalize to the canonical code."""
        cfg = EntsoeConfig.model_validate(
            {"bidding_zone": " pl ", "start_date": "2024-01-01"}
        )
        assert cfg.bidding_zone == "PL"

    @pytest.mark.parametrize("bad", ["   ", None, 123])
    def test_zone_rejects_empty_or_non_string(self, bad):
        """Negative: empty or non-string zones fail as ValidationError."""
        with pytest.raises(ValidationError, match="bidding_zone"):
            EntsoeConfig.model_validate(
                {"bidding_zone": bad, "start_date": "2024-01-01"}
            )


# ---------------------------------------------------------------------------
# Cross-package contract: SupportedMetric vs compute_metrics
# ---------------------------------------------------------------------------
class TestSupportedMetricContract:
    @pytest.mark.parametrize("metric", sorted(SUPPORTED_METRICS))
    def test_compute_metrics_implements_every_supported_metric(self, metric):
        """Contract: compute_metrics produces a value for every Literal member."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 2.0]
        result = compute_metrics(y_true, y_pred, [metric])
        assert metric in result
