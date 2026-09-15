"""
serving/schemas.py — Request/response pydantic models for the HTTP API.

Moved verbatim from ``serve.py`` (Workstream 0 module restructure).
"""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request body for /predict endpoint.

    Accepts a list of feature dictionaries.
    No validation yet — added in Stage 3.
    """
    features: list[dict]


class PredictResponse(BaseModel):
    """Response body for /predict endpoint."""
    predictions: list[float]
