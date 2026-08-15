"""
serve.py — Stage 1: FastAPI serving endpoint for energy price forecasting.

Loads model from MLflow registry at startup and exposes two endpoints:
- GET /health: health check with model version
- POST /predict: accept feature dicts, return predictions

No input validation yet — added in Stage 3.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global config and model (loaded at startup)
cfg = load_config()
model = None
model_version = "unknown"


class PredictRequest(BaseModel):
    """Request body for /predict endpoint.

    Accepts a list of feature dictionaries.
    No validation yet — added in Stage 3.
    """
    features: list[dict]


class PredictResponse(BaseModel):
    """Response body for /predict endpoint."""
    predictions: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model from MLflow registry on startup.

    Prefers the MLFLOW_TRACKING_URI environment variable when set (e.g. the
    API container in docker-compose points at `mlflow:5000`), falling back to
    the value in params.yaml for local runs.

    Args:
        app: The FastAPI application instance.

    Yields:
        None. Control is yielded to the application while it runs.
    """
    global model, model_version, cfg
    logger.info("Loading model from MLflow registry...")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Retry with backoff so a briefly-unavailable MLflow server (e.g. during
    # container startup) doesn't leave the API permanently without a model.
    model_name = cfg["mlflow"]["model_name"]
    max_attempts = 5
    delay_seconds = 2
    for attempt in range(1, max_attempts + 1):
        try:
            model_uri = f"models:/{model_name}/{cfg['serving']['model_stage']}"
            model = mlflow.pyfunc.load_model(model_uri)

            # Get model version from MLflow registry
            client = mlflow.MlflowClient()
            latest_version = client.get_latest_versions(
                model_name, stages=[cfg["serving"]["model_stage"]]
            )
            if latest_version:
                model_version = latest_version[0].version
            logger.info(f"Model loaded successfully (version: {model_version})")
            break
        except Exception as e:
            if attempt < max_attempts:
                logger.warning(
                    f"Model load attempt {attempt}/{max_attempts} failed: {e}. "
                    f"Retrying in {delay_seconds}s..."
                )
                time.sleep(delay_seconds)
            else:
                logger.error(
                    f"Could not load model after {max_attempts} attempts: {e}"
                )
                logger.warning(
                    "API will start but /predict will return 503 until model is available."
                )

    yield

    # Cleanup (if needed)
    logger.info("Shutting down API.")


app = FastAPI(
    title="Energy Price Forecast API",
    description="ML-powered day-ahead electricity price forecasting for Poland",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    if model is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "model_version": model_version,
        }
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": model_version,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict energy prices from feature vectors.

    Accepts a JSON object with a 'features' key containing a list of
    feature dictionaries. Returns a list of predictions.

    Args:
        request: A ``PredictRequest`` containing a list of feature dicts.

    Returns:
        A ``PredictResponse`` with a list of predicted prices.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure MLflow is running and a model is registered.",
        )

    # Convert to DataFrame
    df = pd.DataFrame(request.features)
    predictions = model.predict(df)

    return PredictResponse(predictions=predictions.tolist())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=cfg["serving"]["port"])
