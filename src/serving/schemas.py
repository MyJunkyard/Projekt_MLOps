"""
serving/schemas.py — Request/response pydantic models for the HTTP API.

Moved verbatim from ``serve.py`` (Workstream 0 module restructure).
"""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request body for ``/predict``.

    ``features`` is a list of JSON objects. The serving validator applies
    the model-specific schema and numeric/policy checks after pydantic
    parses this envelope.
    """
    features: list[dict]


class PredictResponse(BaseModel):
    """Response body for /predict endpoint."""
    predictions: list[float]
