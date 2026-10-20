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

#: Metric names understood by ``common.metrics.compute_metrics`` (WS7: the
#: Stage-4 set — ``mape`` removed, ``mase`` opt-in; parent-plan D9 and
#: ``docs/metrics_research.md``). Kept in lock-step with what
#: ``common.metrics`` actually implements by the parametrized contract
#: test ``TestSupportedMetricContract``.
SupportedMetric = Literal["mae", "rmse", "smape", "rmae", "mase", "r2"]

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
    "is_long_weekend",
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

#: ENTSO-E neighbour areas for cross-border day-ahead flows (Workstream 2,
#: EXP-006). Each zone ``Z`` yields a raw ``flow_{z.lower()}_mw`` column at
#: ingest and ``flow_{z.lower()}_lag{L}h`` after availability alignment.
CrossBorderZone = Literal["DE_LU", "CZ", "SK", "LT", "SE_4"]

#: Raw external columns that may carry a real-time publication lag
#: (Workstream 4): the actual load plus one ``{source}_mw`` column per
#: generation source plus one ``flow_{zone}_mw`` column per cross-border
#: neighbour. ``features.availability_lags`` keys must come from this set
#: so a typo fails at config load, not mid-run.
RAW_EXTERNAL_COLUMNS: frozenset[str] = frozenset(
    [
        "load_mw",
        *[f"{source}_mw" for source in get_args(GenerationSource)],
        *[f"flow_{zone.lower()}_mw" for zone in get_args(CrossBorderZone)],
    ]
)

TemporalResolution = Literal["hourly", "daily", "weekly"]

#: Forecast-horizon training strategies (Workstream 1, EXP-014/015).
#: ``single_step`` keeps the Stage-1..3 behaviour: fit ``X_t -> y_t`` and
#: ``temporal.horizon`` is ignored for training; ``direct`` trains one
#: model per horizon step; ``shared`` uses ``horizon_step`` as a feature.
ForecastStrategy = Literal["single_step", "direct", "shared"]

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

    bidding_zone: str = "PL"
    start_date: date = date(2018, 1, 1)
    include_load: bool = True
    allow_synthetic: bool = False

    @field_validator("bidding_zone", mode="before")
    @classmethod
    def _zone_not_empty(cls, v: object) -> str:
        """Normalize the bidding zone code and reject empty values.

        Only a structural guard lives here: require a string, strip
        whitespace, uppercase, require non-empty. Authoritative Area-code
        validation lives in
        ``src.ingestion.entsoe._resolve_bidding_zone`` (offline, against
        the bundled ``entsoe.mappings.Area`` enum) so this leaf model
        stays free of third-party imports and cannot drift from the
        installed ``entsoe-py`` version.

        Args:
            v: Raw ``data.entsoe.bidding_zone`` value from ``params.yaml``.

        Returns:
            The normalized (stripped, uppercased) bidding zone code.

        Raises:
            ValueError: If the zone is not a string or is empty after stripping.
        """
        if not isinstance(v, str):
            raise ValueError(
                "data.entsoe.bidding_zone must be a non-empty string, "
                f"got {v!r}"
            )
        zone = v.strip().upper()
        if not zone:
            raise ValueError("data.entsoe.bidding_zone must be a non-empty string")
        return zone


class DataConfig(_Strict):
    """``data`` — paths, columns, split boundaries, and imputation policy."""

    raw_path: str = "data/raw/"
    superset_dir: str = "data/processed/superset/"
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
    """``temporal`` — data resolution, forecast horizon, and strategy.

    ``strategy`` (Workstream 1) selects how the horizon is trained. No
    cross-field validator is needed: ``strategy != "single_step"``
    requires ``horizon >= 1``, which ``PositiveInt`` already guarantees.
    """

    resolution: TemporalResolution = "hourly"
    horizon: PositiveInt = 24
    strategy: ForecastStrategy = "single_step"

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
        "is_long_weekend",
    ]


class WeatherDerivativesConfig(_Strict):
    """``features.weather.derivatives`` — smoothed weather derivatives (WS3).

    One derivative per configured ``variables`` entry, per ``order``,
    after smoothing with a ``smooth_window`` rolling mean (the same
    smooth-then-differencing recipe as ``features.derivatives`` for the
    target). ``variables`` must be a subset of ``features.weather.variables``
    — validated on ``WeatherConfig`` when the block is enabled, because a
    derivative cannot exist without its base column.
    """

    enabled: bool = False
    variables: list[WeatherVariable] = ["wind_speed_100m"]
    order: list[PositiveInt] = [1]
    smooth_windows: list[PositiveInt] = [3]

    @field_validator("smooth_windows")
    @classmethod
    def _unique_sorted(cls, v: list[int]) -> list[int]:
        """Dedupe and sort: windows are a set, order carries no meaning."""
        return sorted(set(v))


class WeatherConfig(_Strict):
    """``features.weather`` — Open-Meteo weather features (Workstream 3).

    ``allow_synthetic`` gates the synthetic weather fallback explicitly
    (Workstream 3, decision D7): synthetic weather is never generated
    silently — with the default ``false``, a failed download raises, and
    the featurise stage refuses a synthetic-marked cache.
    """

    enabled: bool = False
    variables: list[WeatherVariable] = [
        "temperature_2m",
        "wind_speed_100m",
        "shortwave_radiation",
        "cloud_cover",
        "precipitation",
    ]
    locations: list[str] = ["warsaw"]
    allow_synthetic: bool = False
    derivatives: WeatherDerivativesConfig = WeatherDerivativesConfig()

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

    @model_validator(mode="after")
    def _derivatives_subset_of_variables(self) -> "WeatherConfig":
        """Derivative variables must exist among the weather variables.

        Checked only while the block is enabled: with ``derivatives.enabled:
        false`` the list is inert, so a narrowed ``weather.variables`` must
        not fail at load for an unused block.
        """
        if self.derivatives.enabled:
            unknown = sorted(set(self.derivatives.variables) - set(self.variables))
            if unknown:
                raise ValueError(
                    f"features.weather.derivatives.variables {unknown} not in "
                    f"features.weather.variables {sorted(self.variables)} — "
                    "a derivative cannot exist without its base column"
                )
        return self


class LagsConfig(_Strict):
    """``features.lags`` — lag and rolling-window features.

    ``periods`` are past offsets in native time steps (hours at hourly
    resolution); ``rolling_windows`` are trailing window sizes over which
    mean/std of the target are computed. Both lists are normalized to
    unique sorted values at load time so the generated column order is
    deterministic regardless of YAML order.
    """

    enabled: bool = False
    periods: list[PositiveInt] = [1, 2, 3, 24, 48, 168]
    rolling_windows: list[PositiveInt] = [24, 168]

    @field_validator("periods", "rolling_windows")
    @classmethod
    def _unique_sorted(cls, v: list[int]) -> list[int]:
        """Dedupe and sort: duplicate periods would rebuild the same column."""
        return sorted(set(v))


class DerivativesConfig(_Strict):
    """``features.derivatives`` — smoothed target derivatives (Stage 4)."""

    enabled: bool = False
    order: list[PositiveInt] = [1, 2]
    smooth_windows: list[PositiveInt] = [3, 6, 12]

    @field_validator("smooth_windows")
    @classmethod
    def _unique_sorted(cls, v: list[int]) -> list[int]:
        """Dedupe and sort: windows are a set, order carries no meaning."""
        return sorted(set(v))


class GenerationMixConfig(_Strict):
    """``features.generation_mix`` — ENTSO-E generation by source (WS4).

    The raw ``{source}_mw`` columns are acquired by ingest; the
    availability alignment (``features.availability_lags``) turns them
    into deployment-consistent ``{source}_mw_lag{L}h`` features — the
    raw columns never reach ``features.parquet`` (see
    ``features.main.build_features``).
    """

    enabled: bool = False
    sources: list[GenerationSource] = [
        "wind",
        "solar",
        "coal",
        "gas",
        "nuclear",
        "hydro",
    ]

    @field_validator("sources")
    @classmethod
    def _unique_sorted(cls, v: list[str]) -> list[str]:
        """Dedupe and sort: duplicate sources would rebuild the same column."""
        return sorted(set(v))


class CrossBorderFlowsConfig(_Strict):
    """``features.cross_border_flows`` — ENTSO-E cross-border flows (WS2, EXP-006).

    ``neighbours`` are ENTSO-E Area codes: each ``Z`` yields a raw
    ``flow_{z.lower()}_mw`` column at ingest (WS2 download) and, once
    listed in ``features.availability_lags``, a deployment-consistent
    ``flow_{z.lower()}_lag{L}h`` feature.
    """

    enabled: bool = False
    neighbours: list[CrossBorderZone] = ["DE_LU", "CZ", "SK", "LT", "SE_4"]

    @field_validator("neighbours")
    @classmethod
    def _unique_sorted(cls, v: list[str]) -> list[str]:
        """Dedupe and sort: duplicate zones would rebuild the same column."""
        return sorted(set(v))


class FeatureSelection(_Strict):
    """``features.select`` — the column subset THIS run trains on (WS0).

    Read by training/evaluation only; it never triggers ingest or
    featurise. The default ``features.select: null`` means "all feature
    columns in the materialised superset" (today's behaviour). Schema
    existence of a group/column is checked at train time against the
    superset sidecar (``training/loader.py``), not here — the superset
    may not exist at config-load time.
    """

    groups: list[str] = []
    columns: list[str] = []

    @model_validator(mode="after")
    def _at_least_one_non_empty(self) -> "FeatureSelection":
        """An explicit selection must pick something, with no blank names."""
        blank = [name for name in (*self.groups, *self.columns) if not name.strip()]
        if blank:
            raise ValueError(
                f"features.select entries must be non-empty strings, got {blank!r}"
            )
        if not self.groups and not self.columns:
            raise ValueError(
                "features.select must list at least one group or column "
                "(omit features.select — null — to train on all columns)"
            )
        return self


class FeaturesConfig(_Strict):
    """``features`` — feature-group toggles and their settings.

    Every group defaults to *disabled*, so a minimal config yields no
    engineered features; ``params.yaml`` turns groups on explicitly.

    ``availability_lags`` (Workstream 4) maps each raw external column
    to its real-time publication lag in hours. The alignment block in
    ``build_features`` replaces every configured raw column with
    ``{col}_lag{L}h`` — the value actually known at prediction time —
    so training matches the deployed information set (no leakage).
    """

    calendar: CalendarConfig = CalendarConfig()
    weather: WeatherConfig = WeatherConfig()
    lags: LagsConfig = LagsConfig()
    derivatives: DerivativesConfig = DerivativesConfig()
    generation_mix: GenerationMixConfig = GenerationMixConfig()
    cross_border_flows: CrossBorderFlowsConfig = CrossBorderFlowsConfig()
    availability_lags: dict[str, PositiveInt] = {}
    select: FeatureSelection | None = None

    @field_validator("availability_lags")
    @classmethod
    def _keys_are_known_raw_columns(
        cls, v: dict[str, int]
    ) -> dict[str, int]:
        """Lag keys must be known raw external columns (typo guard)."""
        unknown = sorted(set(v) - RAW_EXTERNAL_COLUMNS)
        if unknown:
            raise ValueError(
                f"features.availability_lags has unknown column(s) "
                f"{unknown} — keys must be raw external columns "
                f"(known: {sorted(RAW_EXTERNAL_COLUMNS)})"
            )
        return v

    @model_validator(mode="after")
    def _generation_sources_covered(self) -> "FeaturesConfig":
        """Enabled externals must have an ``availability_lags`` entry.

        A ``{source}_mw`` or ``flow_{zone}_mw`` column without a lag entry
        would flow into the feature matrix as a current-hour value —
        exactly the deployment leakage this mechanism prevents — so the
        omission fails at load time.
        """
        if self.generation_mix.enabled:
            missing = sorted(
                f"{source}_mw"
                for source in self.generation_mix.sources
                if f"{source}_mw" not in self.availability_lags
            )
            if missing:
                raise ValueError(
                    f"features.generation_mix is enabled but "
                    f"features.availability_lags has no entry for "
                    f"{missing} — every generation source must declare "
                    "its real-time publication lag (>= 1 hour)."
                )
        if self.cross_border_flows.enabled:
            missing_flows = sorted(
                f"flow_{zone.lower()}_mw"
                for zone in self.cross_border_flows.neighbours
                if f"flow_{zone.lower()}_mw" not in self.availability_lags
            )
            if missing_flows:
                raise ValueError(
                    f"features.cross_border_flows is enabled but "
                    f"features.availability_lags has no entry for "
                    f"{missing_flows} — every cross-border flow must declare "
                    "its real-time publication lag (>= 1 hour)."
                )
        return self


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

    primary_metric: SupportedMetric = "mae"
    metrics: list[SupportedMetric] = ["mae", "rmse", "smape", "rmae", "r2"]
    generate_plots: bool = True
    residual_breakdown: list[str] = ["hour", "day_of_week", "month", "is_holiday"]
    # WS4 residual-analysis controls (inert until src/evaluation/analysis.py).
    residual_quantiles: int = 5
    sample_week: date | None = None
    run_analysis_after_train: bool = True
    # WS7 metric plumbing (parent D9 set consumed from step 1 on;
    # naive_period/dm_test/dm_reference_run_id consumed from step 5 on —
    # see docs/stage4_ws7_implementation_plan.md).
    naive_period: PositiveInt = 168
    dm_test: bool = True
    # Fixed DM reference run (WS7 local D8); filled by step 5b per parent D14.
    dm_reference_run_id: str | None = None

    @field_validator("residual_quantiles")
    @classmethod
    def _quantiles_non_negative(cls, v: int) -> int:
        """``0`` disables price-quantile breakdowns; negatives are nonsense."""
        if v < 0:
            raise ValueError(
                f"evaluation.residual_quantiles must be >= 0 (0 disables), got {v}"
            )
        return v

    @model_validator(mode="after")
    def _primary_in_metrics(self) -> "EvaluationConfig":
        """The comparison metric must be among the computed metrics."""
        if self.primary_metric not in self.metrics:
            raise ValueError(
                f"evaluation.primary_metric ({self.primary_metric!r}) must be "
                f"listed in evaluation.metrics ({self.metrics})"
            )
        return self


#: Tags the pipeline assigns itself. A user-supplied ``mlflow.run_tags`` entry
#: with one of these keys never reaches MLflow — see the validator below. Owned
#: here (``config`` is a strict leaf that imports nothing from ``src``) and
#: *imported* by ``training.registry``, not duplicated.
RESERVED_RUN_TAGS = frozenset(
    {
        "stage",
        "params_hash",
        "git_commit",
        "manifest_sha256",
        "superset_path",
        "superset_materialisation_hash",
    }
)


class MlflowConfig(_Strict):
    """``mlflow`` — tracking, experiment, registry, and promotion policy."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "energy-forecast"
    model_name: str = "energy-forecast-model"
    promote_to_production: bool = True
    champion_alias: str = "champion"
    run_tags: dict[str, str] = {}

    @field_validator("run_tags")
    @classmethod
    def _run_tags_not_reserved(cls, v: dict[str, str]) -> dict[str, str]:
        """Reject a ``run_tags`` key the pipeline owns (WS6 D4).

        ``run_tags`` and the pipeline-written tags are **disjoint namespaces**:
        the convention keys (``exp_id``, ``series``) live only in ``run_tags``
        — no code writes them — while the six reserved keys are written only by
        code, so a human entry under the same name can only be a forgery
        attempt or a typo. Rejecting at load time (this module's stated
        philosophy: *"typos fail at load time instead of mid-run"*) turns a
        failure *after a model was trained* into a pre-flight one.

        Raises:
            ValueError: Naming the offending key when the intersection is
                non-empty.
        """
        overlap = sorted(set(v) & RESERVED_RUN_TAGS)
        if overlap:
            raise ValueError(
                f"mlflow.run_tags key(s) {overlap} are reserved for the "
                "pipeline and cannot be set in params.yaml. Reserved keys: "
                f"{sorted(RESERVED_RUN_TAGS)}. Use the convention keys "
                "(e.g. exp_id, series) instead."
            )
        return v


class ServingValidationConfig(_Strict):
    """Serving request limits and generic feature bounds."""

    max_rows: PositiveInt = 1000
    ranges: dict[str, tuple[float, float]] = {}

    @field_validator("ranges")
    @classmethod
    def _valid_ranges(
        cls, value: dict[str, tuple[float, float]]
    ) -> dict[str, tuple[float, float]]:
        for name, bounds in value.items():
            if not name or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(
                    f"invalid serving validation range for {name!r}: {bounds!r}"
                )
        return value


class ServingConfig(_Strict):
    """``serving`` — HTTP API settings and request-validation policy."""

    port: PositiveInt = 8000
    model_alias: str = "champion"
    validation: ServingValidationConfig = ServingValidationConfig()


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
