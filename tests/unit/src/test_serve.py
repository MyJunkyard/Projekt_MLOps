"""
Unit tests for src/serve.py — FastAPI /health and /predict endpoints.
"""

from unittest import mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.serve as serve


@pytest.fixture
def client():
    """TestClient with a clean module state."""
    serve.model = None
    serve.model_version = "unknown"
    return TestClient(serve.app)


class TestHealth:
    def test_health_degraded_when_no_model(self, client):
        """Negative: /health returns degraded when model not loaded."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False

    def test_health_ok_when_model_loaded(self, client):
        """Positive: /health returns ok when model is loaded."""
        serve.model = mock.MagicMock()
        serve.model_version = "1"
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "1"


class TestPredict:
    def test_predict_returns_predictions(self, client):
        """Positive: /predict returns predictions list."""
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([10.0, 20.0, 30.0])
        serve.model = mock_model

        resp = client.post(
            "/predict", json={"features": [{"hour": 1}, {"hour": 2}, {"hour": 3}]}
        )
        assert resp.status_code == 200
        assert resp.json() == {"predictions": [10.0, 20.0, 30.0]}

    def test_predict_503_when_no_model(self, client):
        """Negative: /predict returns 503 when model not loaded."""
        serve.model = None
        resp = client.post("/predict", json={"features": [{"hour": 1}]})
        assert resp.status_code == 503
        assert "Model not loaded" in resp.json()["detail"]