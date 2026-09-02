# Energy Forecast MLOps

MLOps pipeline for day-ahead electricity price forecasting (Poland, ENTSO-E).

## Overview

This project implements a complete MLOps pipeline for forecasting day-ahead electricity prices in Poland using data from ENTSO-E (European Network of Transmission System Operators for Electricity).

### Pipeline Stages

1. **Ingest** — Download raw electricity price data from ENTSO-E API
2. **Featurise** — Engineer calendar, holiday, and lag features
3. **Train** — Train XGBoost model with baseline comparisons
4. **Evaluate** — Compute metrics, generate plots, and analyze residuals
5. **Serve** — Deploy model via FastAPI endpoint

## Prerequisites

- Python >= 3.11
- Docker & Docker Compose (for MLflow tracking server)
- ENTSO-E API key (stored in `secrets/.env`)
- **GNU Make** — required to run `make` commands. Linux/macOS include it by default. **Windows users must install it separately**, e.g. via [Chocolatey](https://chocolatey.org/): `choco install make` (requires admin shell), or [Scoop](https://scoop.sh/): `scoop install make`. Alternatively, you can run the commands directly (see table below).

## Installation

```bash
# Clone the repository
git clone https://github.com/MyJunkyard/Projekt_MLOps.git
cd Projekt_MLOps

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install with development dependencies
pip install -e ".[dev]"
```

## Configuration

Pipeline configuration is managed through `params.yaml`. Key sections:

- `data` — Data source settings, target column, split dates
- `temporal` — Time resolution and gap handling
- `mlflow` — Tracking server URI and model registry
- `evaluation` — Metrics and plot generation settings
- `logging` — Log level configuration

## Usage

### 1. Build and Deploy Containers

The project uses Docker Compose to run three services: **PostgreSQL** (MLflow backend), **MLflow** (tracking server), and **API** (FastAPI model serving).

```bash
# Build images and start PostgreSQL + MLflow in the background
make up
```

This command:
- Builds the MLflow image from `Dockerfile.mlflow`
- Pulls and starts the PostgreSQL 17 container
- Starts the MLflow tracking server (waits for PostgreSQL to be healthy before starting)

**Verify the containers are running:**

```bash
docker compose ps
```

Expected output — both services should show `healthy` or `running`:

```
NAME            IMAGE                          STATUS
energy-postgres postgres:17-alpine             Up X seconds (healthy)
energy-mlflow   projekt_mlops-mlflow           Up X seconds (healthy)
```

**Check MLflow health directly:**

```bash
curl http://localhost:5000/health
```

Once MLflow is healthy, start the API container:

```bash
make serve
```

This builds the API image from `Dockerfile` and starts the FastAPI container. The API loads the model tagged with the `champion` alias from MLflow at startup.

**Verify the API is running:**

```bash
curl http://localhost:8000/health
```

Expected response when a model is registered:

```json
{"status": "ok", "model_loaded": true, "model_version": "1"}
```

If no model is registered yet, the API starts in degraded mode:

```json
{"status": "degraded", "model_loaded": false, "model_version": "unknown"}
```

### 2. What to Do Once Containers Are Ready

#### MLflow UI

Open **http://localhost:5000** in your browser to access the MLflow tracking UI. Here you can:

- View all experiment runs and their metrics (RMSE, MAE, MAPE, R²)
- Compare runs side-by-side
- Inspect artifacts (evaluation plots, residual breakdowns)
- Manage the model registry (view versions, aliases, and stages)

#### API Endpoints

Once a model is trained and registered (see below), the API at **http://localhost:8000** exposes:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check — confirms model is loaded and shows version |
| POST | `/predict` | Send feature vectors, receive price predictions |
| GET | `/docs` | Interactive Swagger UI (auto-generated) |
| GET | `/redoc` | Alternative API documentation (ReDoc) |

**Example prediction request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [{"hour": 12, "day_of_week": 3, "month": 6, "price_lag_1": 85.2}]}'
```

#### Run the Pipeline

With MLflow running in Docker, execute the pipeline locally (requires activated virtual environment):

```bash
make ingest      # Download raw data from ENTSO-E API
make featurise   # Engineer calendar, holiday, and lag features
make train       # Train XGBoost model, log to MLflow, register as champion
make evaluate    # Compute metrics, generate plots
```

Or run the full pipeline in one command:

```bash
make all
```

After training, the model is automatically registered in MLflow with the `champion` alias. The API will load it on next restart (or immediately if already running, since it retries on startup).

#### Stop Containers

```bash
make down
```

This stops and removes all containers but preserves data (PostgreSQL data and MLflow artifacts persist in Docker volumes).

To also remove stored data:

```bash
docker compose down -v
```

### Run Tests

```bash
make test
# or directly:
pytest tests/ -v
```

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make up` | Build images and start PostgreSQL + MLflow containers |
| `make serve` | Build and start the API container |
| `make down` | Stop all containers |
| `make ingest` | Download raw data from ENTSO-E |
| `make featurise` | Engineer features |
| `make train` | Train model and register in MLflow |
| `make evaluate` | Evaluate model and generate plots |
| `make all` | Run full pipeline (ingest → featurise → train → evaluate) |
| `make test` | Run test suite |
| `make compare` | Print MLflow runs for comparison |
| `make clean` | Remove processed and reference data files |

## Project Structure

```
.
├── data/
│   ├── raw/          # Raw downloaded data
│   ├── processed/    # Featurised data
│   └── reference/    # Reference data for drift detection
├── docs/             # Documentation
├── models/           # Saved models
├── reports/          # Evaluation plots and reports
├── src/              # Source code
│   ├── ingest.py     # Data ingestion
│   ├── featurise.py  # Feature engineering
│   ├── train.py      # Model training
│   ├── evaluate.py   # Model evaluation
│   ├── serve.py      # API serving
│   ├── monitor.py    # Drift monitoring
│   └── utils.py      # Shared utilities
├── tests/            # Test suite
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── functional/   # End-to-end tests
├── params.yaml       # Pipeline configuration
├── pyproject.toml    # Project metadata and dependencies
├── Makefile          # Build automation
└── requirements.txt  # Generated from pyproject.toml
```

## License

MIT