"""
ingestion/weather/locations.py — Weather location registry.

The single source of truth for the location → coordinates mapping,
shared by the Open-Meteo client (real data) and the synthetic generator
(neither should hardcode its own list). Workstream 3, decisions D2/D8.
"""

LOCATION_COORDS: dict[str, tuple[float, float]] = {
    "warsaw": (52.23, 21.01),
    "krakow": (50.06, 19.94),
    "gdansk": (54.35, 18.65),
    "wroclaw": (51.11, 17.11),
}


def validate_location(location: str) -> tuple[float, float]:
    """Resolve a location name to its coordinates.

    Args:
        location: Location key (must be in ``LOCATION_COORDS``).

    Returns:
        The ``(latitude, longitude)`` pair.

    Raises:
        ValueError: If the location is unknown (lists the valid set).
    """
    if location not in LOCATION_COORDS:
        raise ValueError(
            f"Unknown weather location {location!r}; valid locations: "
            f"{sorted(LOCATION_COORDS)}"
        )
    return LOCATION_COORDS[location]
