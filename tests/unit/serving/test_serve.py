"""
Unit tests for the serving package — FastAPI /health and /predict endpoints.
"""

from unittest import mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.serving.app as serve
from src.common.schema import ColumnRole, ColumnSpec, FeatureSchema


@pytest.fixture
def client():
    """TestClient with a clean module state."""
    serve.model = None
    serve.model_version = "unknown"
    serve.feature_schema = None
    return TestClient(serve.app)


@pytest.fixture
def prediction_schema():
    return FeatureSchema(
        columns={
            "hour": ColumnSpec(name="hour", role=ColumnRole.FEATURE),
            "is_holiday": ColumnSpec(name="is_holiday", role=ColumnRole.FEATURE),
        }
    )


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
    def test_predict_returns_predictions(self, client, prediction_schema):
        serve.feature_schema = prediction_schema
        serve.cfg.serving.validation.ranges["hour"] = (0, 23)
        """Positive: /predict returns predictions list."""
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([10.0, 20.0, 30.0])
        serve.model = mock_model

        resp = client.post(
            "/predict",
            json={
                "features": [
                    {"hour": 1, "is_holiday": 0},
                    {"hour": 2, "is_holiday": 0},
                    {"hour": 3, "is_holiday": 1},
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.json() == {"predictions": [10.0, 20.0, 30.0]}
        received = mock_model.predict.call_args.args[0]
        assert list(received.columns) == ["hour", "is_holiday"]

    def test_predict_missing_column_returns_422(self, client, prediction_schema):
        serve.feature_schema = prediction_schema
        serve.model = mock.MagicMock()
        resp = client.post("/predict", json={"features": [{"hour": 1}]})
        assert resp.status_code == 422
        assert "missing" in resp.json()["detail"]
        serve.model.predict.assert_not_called()

    def test_predict_nan_returns_422(self, client, prediction_schema):
        serve.feature_schema = prediction_schema
        serve.model = mock.MagicMock()
        response = client.post(
            "/predict",
            content='{"features": [{"hour": NaN, "is_holiday": 0}]}',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422

    def test_predict_oversized_batch_returns_422(self, client, prediction_schema):
        serve.feature_schema = prediction_schema
        serve.model = mock.MagicMock()
        serve.cfg.serving.validation.max_rows = 1
        resp = client.post(
            "/predict",
            json={
                "features": [
                    {"hour": 1, "is_holiday": 0},
                    {"hour": 2, "is_holiday": 0},
                ]
            },
        )
        assert resp.status_code == 422
        assert "maximum" in resp.json()["detail"]

    def test_predict_range_violation_returns_422(self, client, prediction_schema):
        serve.feature_schema = prediction_schema
        serve.model = mock.MagicMock()
        serve.cfg.serving.validation.ranges["hour"] = (0, 23)
        resp = client.post(
            "/predict", json={"features": [{"hour": 24, "is_holiday": 0}]}
        )
        assert resp.status_code == 422
        assert "outside" in resp.json()["detail"]

    def test_missing_schema_is_permissive(self, client):
        serve.feature_schema = None
        serve.model = mock.MagicMock()
        serve.model.predict.return_value = np.array([10.0])
        resp = client.post("/predict", json={"features": [{"hour": 1}]})
        assert resp.status_code == 200
        assert resp.json() == {"predictions": [10.0]}

    def test_predict_503_when_no_model(self, client):
        """Negative: /predict returns 503 when model not loaded."""
        serve.model = None
        resp = client.post("/predict", json={"features": [{"hour": 1}]})
        assert resp.status_code == 503
        assert "Model not loaded" in resp.json()["detail"]
