"""
Unit tests for the config package — PipelineConfig schema validation
and the params.yaml loader.

Validation must fail at load time, before any pipeline stage runs.
"""

from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from src.common.metrics import compute_metrics
from src.config import load_config
from src.config.models import (
    SUPPORTED_METRICS,
    DataConfig,
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
