"""
features/calendar.py — Calendar and holiday features.

Extracted verbatim from ``featurise.py`` (Workstream 0 module restructure).
"""

import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame, holidays: pd.DatetimeIndex
) -> pd.DataFrame:
    """Add calendar-based features from timestamp column.

    Features added:
    - hour: 0-23
    - day_of_week: 0=Monday, 6=Sunday
    - month: 1-12
    - week_of_year: 1-53
    - is_holiday: True for Polish public holidays
    - is_workday: not weekend AND not holiday

    Args:
        df: DataFrame with a ``timestamp`` datetime column (the
            featurisation input contract: ``timestamp``, target column
            from ingest).
        holidays: Sorted holiday midnights (UTC) from ``_get_holiday_dates``.

    Returns:
        The same DataFrame (mutated in place and returned) with the
        calendar feature columns ``hour``, ``day_of_week``, ``month``,
        ``week_of_year``, ``is_holiday``, ``is_workday`` added.
    """
    timestamp = df["timestamp"]
    df["hour"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["month"] = timestamp.dt.month
    df["week_of_year"] = timestamp.dt.isocalendar().week.astype(int)

    # Polish public holidays
    df = add_holiday_features(df, holidays)

    # Workday = weekday (Mon-Fri) and not a holiday
    df["is_workday"] = (df["day_of_week"] < 5) & (~df["is_holiday"])

    return df


def _get_holiday_dates(df: pd.DataFrame, country: str = "Poland") -> pd.DatetimeIndex:
    """Build a sorted DatetimeIndex of public holiday midnights (UTC).

    The calendar covers the data's year range padded by one year on each
    side, so proximity features are correct for timestamps at the very
    start/end of the data (e.g. Dec 31 needs the next year's New Year).

    This is the single place where the holiday calendar is built; all
    holiday feature functions receive it precomputed (review point 6).

    Args:
        df: DataFrame with a ``timestamp`` datetime column (its year
            range, padded ±1 year, defines the calendar span).
        country: Country name for the holiday calendar (default: "Poland").

    Returns:
        Sorted, deduplicated ``DatetimeIndex`` of holiday midnights (UTC).

    Raises:
        ValueError: If the country has no supported holiday calendar.
    """
    from workalendar.europe import Poland

    calendars = {"Poland": Poland}
    if country not in calendars:
        raise ValueError(f"Unsupported holiday calendar country: {country!r}")
    cal = calendars[country]()

    years = df["timestamp"].dt.year
    holiday_dates: set = set()
    for year in range(years.min() - 1, years.max() + 2):
        for date, _ in cal.holidays(year):
            holiday_dates.add(date)

    return pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in sorted(holiday_dates)])


def add_holiday_features(df: pd.DataFrame, holidays: pd.DatetimeIndex) -> pd.DataFrame:
    """Add a public holiday flag using a precomputed holiday calendar.

    Holidays include: New Year, Easter Monday, Labour Day, Constitution Day,
    Corpus Christi, Assumption, All Saints, Independence Day, Christmas Day,
    and Second Day of Christmas.

    Args:
        df: DataFrame with a ``timestamp`` datetime column.
        holidays: Sorted holiday midnights (UTC) from ``_get_holiday_dates``.

    Returns:
        The same DataFrame (mutated in place and returned) with the
        boolean ``is_holiday`` column added.
    """
    df["is_holiday"] = df["timestamp"].dt.normalize().isin(holidays)

    return df


def add_holiday_proximity_features(
    df: pd.DataFrame, holidays: pd.DatetimeIndex
) -> pd.DataFrame:
    """Add holiday proximity features using a precomputed holiday calendar.

    Features added:
    - days_to_next_holiday: integer days until the next upcoming holiday
    - days_since_last_holiday: integer days since the most recent past holiday

    Both are computed on whole calendar days, so every hour of a day gets
    the same value (e.g. 23:00 on the day before a holiday gives 1, not 0 —
    review point 5). On a holiday itself both are 0. If there is no holiday
    in range at all, the value is NaN rather than an ambiguous 0.

    These capture price spikes that often appear in the days before a long
    weekend.

    Args:
        df: DataFrame with a ``timestamp`` datetime column.
        holidays: Sorted holiday midnights (UTC) from ``_get_holiday_dates``.

    Returns:
        The same DataFrame (mutated in place and returned) with nullable
        integer columns ``days_to_next_holiday`` and
        ``days_since_last_holiday`` added (NaN when no holiday is in
        range).
    """
    # Whole calendar days as proleptic Gregorian ordinals — immune to the
    # Timedelta.days truncation that affects sub-day remainders.
    ts_days = np.array(
        [d.toordinal() for d in df["timestamp"].dt.date], dtype=np.int64
    )
    holiday_days = np.array([d.toordinal() for d in holidays.date], dtype=np.int64)

    # Vectorised binary search: O(n log m) instead of a full scan per
    # timestamp (review point 5). side="right" - 1 gives the last holiday
    # on or before ts; side="left" gives the first holiday on or after ts.
    # On a holiday both point at the holiday itself, yielding 0 by
    # construction.
    last_idx = np.searchsorted(holiday_days, ts_days, side="right") - 1
    next_idx = np.searchsorted(holiday_days, ts_days, side="left")

    days_since = np.full(len(df), np.nan)
    has_last = last_idx >= 0
    days_since[has_last] = ts_days[has_last] - holiday_days[last_idx[has_last]]

    days_to_next = np.full(len(df), np.nan)
    has_next = next_idx < len(holiday_days)
    days_to_next[has_next] = holiday_days[next_idx[has_next]] - ts_days[has_next]

    df["days_to_next_holiday"] = pd.Series(days_to_next, index=df.index).astype(
        "Int64"
    )
    df["days_since_last_holiday"] = pd.Series(days_since, index=df.index).astype(
        "Int64"
    )

    return df
