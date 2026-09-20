"""
Unit tests for feature-name construction/parsing (``features/naming.py``).

The round-trip property tests are the anti-fragility guarantee: as long
as ``make ↔ parse`` agree for every constructed name, the single home
for name strings cannot drift away from its documented conventions.
"""

import pytest

from src.features.naming import (
    make_availability_lag_name,
    make_lag_name,
    make_rolling_name,
    parse_availability_lag_name,
    parse_lag_name,
    parse_rolling_name,
)


class TestLagNames:
    @pytest.mark.parametrize("period", [1, 2, 3, 24, 48, 168])
    def test_round_trip(self, period):
        name = make_lag_name(period)
        assert parse_lag_name(name) == period

    def test_parse_rejects_non_lag(self):
        with pytest.raises(ValueError, match="lag"):
            parse_lag_name("rolling_mean_24h")

    def test_parse_rejects_availability_lag_form(self):
        """The two mechanisms stay distinguishable: load_mw_lag1h is NOT lag_."""
        with pytest.raises(ValueError):
            parse_lag_name("load_mw_lag1h")


class TestRollingNames:
    @pytest.mark.parametrize("stat", ["mean", "std"])
    @pytest.mark.parametrize("window", [24, 168])
    def test_round_trip(self, stat, window):
        name = make_rolling_name(stat, window)
        assert parse_rolling_name(name) == (stat, window)

    def test_invalid_stat_rejected_at_construction(self):
        with pytest.raises(ValueError, match="mean.*std"):
            make_rolling_name("median", 24)

    def test_parse_rejects_non_rolling(self):
        with pytest.raises(ValueError):
            parse_rolling_name("lag_24h")


class TestAvailabilityLagNames:
    @pytest.mark.parametrize(
        "col,lag",
        [("load_mw", 1), ("wind_mw", 2), ("solar_mw", 24)],
    )
    def test_round_trip(self, col, lag):
        name = make_availability_lag_name(col, lag)
        assert parse_availability_lag_name(name) == (col, lag)

    def test_name_format(self):
        assert make_availability_lag_name("load_mw", 1) == "load_mw_lag1h"

    def test_zero_lag_rejected(self):
        """A 0-hour 'lag' is the raw leaky column — construction fails fast."""
        with pytest.raises(ValueError, match=">= 1"):
            make_availability_lag_name("load_mw", 0)

    def test_parse_rejects_plain_name(self):
        with pytest.raises(ValueError):
            parse_availability_lag_name("load_mw")

    def test_parse_rejects_target_lag_form(self):
        """lag_24h is not an availability-lag name (prefix required)."""
        with pytest.raises(ValueError):
            parse_availability_lag_name("lag_24h")
