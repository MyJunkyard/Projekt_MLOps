"""
config/models.py — Pydantic schema of ``params.yaml``.

The single source of truth for the configuration *shape*: every key in
``params.yaml`` has a typed field here, and unknown keys are rejected
(``extra="forbid"``) so typos fail at load time instead of mid-run.
``params.yaml`` remains the source of truth for the *values*.

Layering rule: this module is a strict leaf — it must import nothing
from ``src`` (only stdlib + pydantic). Every other package may depend on
it; nothing it depends on can depend back, so import cycles are
structurally impossible.

Design notes:

- **Required vs defaulted fields.** Only fields with no sensible default
  are required (``data.target_col``, ``data.train_end``, ``data.val_end``,
  ``model.type``); everything else defaults to the current effective
  value, preserving the previous ``.get(key, default)`` fallback
  behaviours (e.g. a missing ``mlflow.promote_to_production`` still means
  "promotion enabled").
- **Dates are ``datetime.date``.** Pydantic parses the YAML strings once
  at load time; ``pd.Timestamp(d, tz="UTC")`` and entsoe-py both accept
  ``date`` objects directly.
- **Vocabulary Literals live here** (``SupportedMetric``,
  ``CalendarFeature``, ...). They are the contract the validators and the
  pipeline code share; a parametrized test pins
  ``SupportedMetric`` to what ``common.metrics.compute_metrics``
  actually implements.
- **Sub-config signatures.** Consumer functions take the narrowest
  sub-model they consume (``DataConfig``, ``ModelConfig``, ...) rather
  than the whole ``PipelineConfig`` — dependencies stay explicit.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveInt,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Vocabulary (shared contracts, defined here so this module stays a leaf)
# ---------------------------------------------------------------------------

#: Metric names understood by ``common.metrics.compute_metrics``.
SupportedMetric = Literal["rmse", "mae", "mape", "r2"]

SUPPORTED_METRICS: frozenset[str] = frozenset(get_args(SupportedMetric))

#: Calendar feature names produced by ``features.calendar``.
CalendarFeature = Literal[
    "hour",
    "day_of_week",
    "month",
    "week_of_year",
    "is_holiday",
    "is_workday",
    "days_to_next_holiday",
    "days_since_last_holiday",
]

#: Open-Meteo hourly weather variables (Workstream 3).
WeatherVariable = Literal[
    "temperature_2m",
    "wind_speed_100m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
]

#: ENTSO-E generation sources mapped to ``*_mw`` columns (Workstream 4).
GenerationSource = Literal["wind", "solar", "coal", "gas", "nuclear", "hydro"]

TemporalResolution = Literal["hourly", "daily", "weekly"]

FillMethod = Literal["ffill", "interpolate"]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class _Strict(BaseModel):
    """Base for all config models: unknown keys are rejected.

    A typo in ``params.yaml`` (e.g. ``raw_pat``) must fail validation at
    load time, not silently disappear. Adding a key to ``params.yaml``
    therefore requires adding the field here in the same change — the
    model is the source of truth for the configuration shape.
    """

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Section models (order follows params.yaml, top to bottom)
# ---------------------------------------------------------------------------


class EntsoeConfig(_Strict):
    """``data.entsoe`` — ENTSO-E Transparency Platform download settings."""

    bidding_zone: str = "PSE"
    start_date: date = date(2018, 1, 1)
    include_load: bool = True


class DataConfig(_Strict):
    """``data`` — paths, columns, split boundaries, and imputation policy."""

    raw_path: str = "data/raw/"
    processed_path: str = "data/processed/features.parquet"
    reference_path: str = "data/reference/reference.parquet"
    target_col: str
    date_col: str = "timestamp"
    train_end: date
    val_end: date
    # test set = everything after val_end
    max_gap_periods: PositiveInt = 2
    fill_method: FillMethod = "ffill"
    add_is_imputed_flag: bool = True
    drop_long_gaps: bool = True
    entsoe: EntsoeConfig = EntsoeConfig()

    @model_validator(mode="after")
    def _split_dates_ordered(self) -> "DataConfig":
        """Fail at load time when the train/val boundaries are inverted."""
        if self.val_end <= self.train_end:
            raise ValueError(
                f"data.val_end ({self.val_end}) must be after "
                f"data.train_end ({self.train_end})"
            )
        return self


class TemporalConfig(_Strict):
    """``temporal`` — data resolution and forecast horizon."""

    resolution: TemporalResolution = "hourly"
    horizon: PositiveInt = 24

    @property
    def pandas_freq(self) -> str:
        """Pandas frequency alias for the resolution (``"h"``, ``"D"``, ``"W"``)."""
        return {"hourly": "h", "daily": "D", "weekly": "W"}[self.resolution]


class CalendarConfig(_Strict):
    """``features.calendar`` — calendar and holiday features."""

    enabled: bool = False
    include: list[CalendarFeature] = [
        "hour",
        "day_of_week",
        "month",
        "week_of_year",
        "is_holiday",
        "is_workday",
        "days_to_next_holiday",
        "days_since_last_holiday",
    ]


class WeatherConfig(_Strict):
    """``features.weather`` — Open-Meteo weather features (Workstream 3)."""

    enabled: bool = False
    variables: list[WeatherVariable] = [
        "temperature_2m",
        "wind_speed_100m",
        "shortwave_radiation",
        "cloud_cover",
    ]
    locations: list[str] = ["warsaw"]

    @field_validator("locations")
    @classmethod
    def _locations_non_empty_lowercase(cls, v: list[str]) -> list[str]:
        """Locations are cache-file keys: non-empty, lowercase, unique."""
        if not v:
            raise ValueError("features.weather.locations must not be empty")
        for loc in v:
            if not loc or loc != loc.lower():
                raise ValueError(
                    f"features.weather.locations entries must be lowercase "
                    f"non-empty strings, got {loc!r}"
                )
        if len(set(v)) != len(v):
            raise ValueError("features.weather.locations contains duplicates")
        return v


class LagsConfig(_Strict):
    """``features.lags`` — lag and rolling-window features.

    ``rolling_windows`` is pre-provisioned for Workstream 2 (which adds
    the key to ``params.yaml``); ``None`` keeps the current hardcoded
    ``[24, 168]`` default in ``features.lags.add_rolling_features``.
    """

    enabled: bool = False
    periods: list[PositiveInt] = [1, 2, 3, 24, 48, 168]
    rolling_windows: list[PositiveInt] | None = None


class DerivativesConfig(_Strict):
    """``features.derivatives`` — smoothed target derivatives (Stage 4)."""

    enabled: bool = False
    order: list[PositiveInt] = [1, 2]
    smooth_window: PositiveInt = 3


class GenerationMixConfig(_Strict):
    """``features.generation_mix`` — ENTSO-E generation by source (WS4)."""

    enabled: bool = False
    sources: list[GenerationSource] = [
        "wind",
        "solar",
        "coal",
        "gas",
        "nuclear",
        "hydro",
    ]


class CrossBorderFlowsConfig(_Strict):
    """``features.cross_border_flows`` — placeholder (Stage 4+)."""

    enabled: bool = False


class FeaturesConfig(_Strict):
    """``features`` — feature-group toggles and their settings.

    Every group defaults to *disabled*, so a minimal config yields no
    engineered features; ``params.yaml`` turns groups on explicitly.
    """

    calendar: CalendarConfig = CalendarConfig()
    weather: WeatherConfig = WeatherConfig()
    lags: LagsConfig = LagsConfig()
    derivatives: DerivativesConfig = DerivativesConfig()
    generation_mix: GenerationMixConfig = GenerationMixConfig()
    cross_border_flows: CrossBorderFlowsConfig = CrossBorderFlowsConfig()


class ModelConfig(_Strict):
    """``model`` — model class (dotted import path) and hyperparameters."""

    type: str
    params: dict[str, Any] = {}

    @field_validator("type")
    @classmethod
    def _dotted_path(cls, v: str) -> str:
        """``model.type`` must be a dotted import path (at least one dot)."""
        parts = v.split(".")
        if len(parts) < 2 or not all(
            part and part.isidentifier() for part in parts
        ):
            raise ValueError(
                f"model.type must be a dotted import path like "
                f"'xgboost.XGBRegressor', got {v!r}"
            )
        return v


class EvaluationConfig(_Strict):
    """``evaluation`` — metrics, plots, and residual breakdown columns."""

    primary_metric: SupportedMetric = "rmse"
    metrics: list[SupportedMetric] = ["rmse", "mae", "mape", "r2"]
    generate_plots: bool = True
    residual_breakdown: list[str] = ["hour", "day_of_week", "month", "is_holiday"]

    @model_validator(mode="after")
    def _primary_in_metrics(self) -> "EvaluationConfig":
        """The comparison metric must be among the computed metrics."""
        if self.primary_metric not in self.metrics:
            raise ValueError(
                f"evaluation.primary_metric ({self.primary_metric!r}) must be "
                f"listed in evaluation.metrics ({self.metrics})"
            )
        return self


class MlflowConfig(_Strict):
    """``mlflow`` — tracking, experiment, registry, and promotion policy."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "energy-forecast"
    model_name: str = "energy-forecast-model"
    promote_to_production: bool = True
    champion_alias: str = "champion"
    run_tags: dict[str, str] = {}


class ServingConfig(_Strict):
    """``serving`` — HTTP API settings (deep validation arrives in WS6b)."""

    port: PositiveInt = 8000
    model_alias: str = "champion"


class LoggingConfig(_Strict):
    """``logging`` — level and optional log-file override."""

    level: LogLevel = "INFO"
    file: str | None = None

    @field_validator("level", mode="before")
    @classmethod
    def _level_case_insensitive(cls, v: object) -> object:
        """Accept ``debug``/``Debug``/... — normalize to the upper-case form."""
        if isinstance(v, str):
            return v.upper()
        return v


class PipelineConfig(_Strict):
    """Root of the validated configuration (mirrors ``params.yaml``)."""

    data: DataConfig
    temporal: TemporalConfig = TemporalConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig
    evaluation: EvaluationConfig = EvaluationConfig()
    mlflow: MlflowConfig = MlflowConfig()
    serving: ServingConfig = ServingConfig()
    logging: LoggingConfig = LoggingConfig()
