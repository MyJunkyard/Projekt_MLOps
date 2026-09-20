"""
ingestion/weather/naming.py — Weather feature column naming contract.

Weather columns are always named ``{location}__{variable}`` (e.g.
``warsaw__temperature_2m``), Workstream 3 decision D9:

- the double underscore is unambiguous to parse — Open-Meteo variable
  names contain only single underscores;
- filtering by location is a prefix check, by variable a suffix check;
- location-first groups columns contiguously (readable in
  ``df.columns``, feature importances, logs);
- the location is *always* in the name, even with a single location, so
  the feature schema stays stable when more locations are added.

Flat strings are a hard requirement: XGBoost, MLflow, the serving API,
and ``feature_schema.json`` all assume string feature names (MultiIndex
columns were rejected — decision D9).
"""

import pandas as pd


def weather_feature_name(variable: str, location: str) -> str:
    """Return the pipeline-wide weather column name.

    Args:
        variable: Open-Meteo variable name (e.g. ``temperature_2m``).
        location: Location key from ``weather.locations.LOCATION_COORDS``.

    Returns:
        ``"{location}__{variable}"`` (e.g. ``warsaw__temperature_2m``).
    """
    return f"{location}__{variable}"


def parse_weather_feature_name(name: str) -> tuple[str, str]:
    """Split a weather column name into ``(location, variable)``.

    Args:
        name: A column name produced by :func:`weather_feature_name`.

    Returns:
        The ``(location, variable)`` pair, e.g.
        ``("warsaw", "temperature_2m")``.

    Raises:
        ValueError: If ``name`` does not follow the
            ``{location}__{variable}`` convention.
    """
    location, separator, variable = name.partition("__")
    if not separator or not location or not variable or "__" in variable:
        raise ValueError(
            f"{name!r} is not a weather feature column — expected the "
            f"'{{location}}__{{variable}}' convention (e.g. "
            f"'warsaw__temperature_2m')"
        )
    return location, variable


def select_weather_columns(
    df: pd.DataFrame,
    location: str | None = None,
    variable: str | None = None,
) -> list[str]:
    """Select weather columns from a DataFrame by location and/or variable.

    Non-weather columns (names that do not parse as
    ``{location}__{variable}``) are ignored, so this is safe to call on
    any pipeline frame.

    Args:
        df: DataFrame whose columns are inspected.
        location: Restrict to this location (``None`` = any).
        variable: Restrict to this variable (``None`` = any).

    Returns:
        Matching column names, in the DataFrame's column order.
    """
    selected: list[str] = []
    for column in df.columns:
        try:
            column_location, column_variable = parse_weather_feature_name(
                str(column)
            )
        except ValueError:
            continue
        if location is not None and column_location != location:
            continue
        if variable is not None and column_variable != variable:
            continue
        selected.append(str(column))
    return selected
