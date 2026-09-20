"""
features/naming.py — Single home for feature-name construction/parsing.

Every feature-name string in the project is built here and only here.
Blocks call the ``make_*`` constructors; tests (and only tests) may use
the ``parse_*`` round-trip helpers. Consumers must never regex column
names — they query the :class:`~src.common.schema.FeatureSchema`
metadata instead. The parse helpers exist so a property test can pin
``make ↔ parse`` agreement, keeping the two sides from drifting apart.

Conventions (pinned by ``tests/unit/features/test_naming.py``):

- ``lag_{p}h`` — target lag, p hours back.
- ``rolling_{stat}_{w}h`` — trailing rolling statistic of the target.
- ``{col}_lag{L}h`` — availability-lagged external column (e.g.
  ``load_mw_lag1h``): the value of ``col`` as it was known L hours
  before the row's timestamp. Distinct from ``lag_{p}h`` (target lags)
  so the two mechanisms stay distinguishable in contracts.
- ``{location}__{variable}`` — weather (Workstream 3, decision D9;
  kept in ``ingestion/weather/naming.py`` — this module does not
  duplicate it).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Target lag / rolling names (existing WS2 convention, now centralized)
# ---------------------------------------------------------------------------

_LAG_RE = re.compile(r"^lag_(\d+)h$")
_ROLLING_RE = re.compile(r"^rolling_(mean|std)_(\d+)h$")


def make_lag_name(period: int) -> str:
    """``lag_{p}h`` — target lag name."""
    return f"lag_{period}h"


def parse_lag_name(name: str) -> int:
    """Parse ``lag_{p}h`` back to ``p``.

    Raises:
        ValueError: If the name does not match the pattern.
    """
    match = _LAG_RE.match(name)
    if not match:
        raise ValueError(f"{name!r} is not a 'lag_<p>h'-style name")
    return int(match.group(1))


def make_rolling_name(stat: str, window: int) -> str:
    """``rolling_{stat}_{w}h`` — rolling statistic name (stat ∈ {mean, std})."""
    if stat not in ("mean", "std"):
        raise ValueError(f"stat must be 'mean' or 'std', got {stat!r}")
    return f"rolling_{stat}_{window}h"


def parse_rolling_name(name: str) -> tuple[str, int]:
    """Parse ``rolling_{stat}_{w}h`` back to ``(stat, w)``.

    Raises:
        ValueError: If the name does not match the pattern.
    """
    match = _ROLLING_RE.match(name)
    if not match:
        raise ValueError(f"{name!r} is not a 'rolling_<stat>_<w>h'-style name")
    return match.group(1), int(match.group(2))


# ---------------------------------------------------------------------------
# Availability-lagged external columns (WS4)
# ---------------------------------------------------------------------------

_AVAILABILITY_LAG_RE = re.compile(r"^(.+)_lag(\d+)h$")


def make_availability_lag_name(col: str, lag_hours: int) -> str:
    """``{col}_lag{L}h`` — availability-lagged external column name.

    Args:
        col: Raw external column name (e.g. ``load_mw``, ``wind_mw``).
        lag_hours: Publication lag in hours (≥ 1).

    Raises:
        ValueError: If ``lag_hours`` < 1 (a 0-hour "lag" would be the raw
            column itself — the leaky form this mechanism exists to
            prevent).
    """
    if lag_hours < 1:
        raise ValueError(
            f"availability lag must be >= 1 hour, got {lag_hours} "
            f"(for {col!r})"
        )
    return f"{col}_lag{lag_hours}h"


def parse_availability_lag_name(name: str) -> tuple[str, int]:
    """Parse ``{col}_lag{L}h`` back to ``(col, L)``.

    Raises:
        ValueError: If the name does not match the pattern.
    """
    match = _AVAILABILITY_LAG_RE.match(name)
    if not match:
        raise ValueError(
            f"{name!r} is not a '<col>_lag<L>h'-style name"
        )
    return match.group(1), int(match.group(2))
