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
- ENTSO-E API key — `ENTSOE_API_KEY` env var for local runs, or
  `secrets/entsoe_api_token.txt` (mounted as `ENTSOE_API_TOKEN_FILE`)
  for the Docker `api` service. Configure exactly one source.
  Without a key, `make ingest` aborts unless
  `data.entsoe.allow_synthetic: true` is set explicitly (offline dev/CI
  only — never production training). Local example (PowerShell):
  `$env:ENTSOE_API_KEY = (Get-Content secrets/entsoe_api_token.txt -Raw).Trim()`
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

### Known data notes

- **ENTSO-E currency switch**: PL day-ahead prices were published in **PLN**
  for delivery days up to 2019-11-19 and in **EUR** from 2019-11-20 onward
  (the API response declares the unit in `<currency_Unit.name>`; entsoe-py
  performs no FX conversion). `data.entsoe.start_date` is therefore set to
  `2019-11-20` so `price_eur_mwh` is always EUR. *Possible follow-up*:
  convert the 2018-01-01..2019-11-19 era with ECB reference exchange rates
  and move `start_date` back to recover ~22 months of history.
- **No `nuclear` in the generation mix**: Poland has no operating nuclear
  plants, so ENTSO-E has no matching PSR type — the column would be all-NaN
  and, with `data.drop_long_gaps: true`, would drop every row at ingest.
  `nuclear_mw` is therefore absent from `features.generation_mix.sources`
  and `features.availability_lags`.

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

**Rebuilding images after Dockerfile changes:**

```bash
make rebuild          # rebuild all images and restart the stack
make rebuild-mlflow   # rebuild only the MLflow tracking server
make rebuild-api      # rebuild only the API
make rebuild-fresh    # rebuild all images without cache
```

> **Known symptom:** if the MLflow tracking API returns HTTP 500 while
> `GET /health` still returns 200, the running image predates the psycopg3
> dependency fix in `Dockerfile.mlflow` (SQLAlchemy ≥ 2.1 resolves
> `postgresql://` URLs to the psycopg3 dialect) — `make rebuild-mlflow`
> resolves it.

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

Schema-backed requests must contain **exactly** the champion model's
`FEATURE` columns (24 as of the current champion — the authoritative list is
`data/processed/features_schema.json`, mirrored in MLflow); missing or extra
keys return HTTP 422. An optional `timestamp` key must be timezone-aware.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [{
    "hour": 12, "day_of_week": 3, "month": 6, "week_of_year": 24,
    "is_holiday": 0, "is_workday": 1,
    "days_to_next_holiday": 5, "days_since_last_holiday": 3,
    "coal_mw_lag1h": 5200, "gas_mw_lag1h": 1800, "hydro_mw_lag1h": 300,
    "load_mw_lag1h": 15400, "solar_mw_lag1h": 4100, "wind_mw_lag1h": 6900,
    "lag_1h": 82.4, "lag_2h": 79.1, "lag_3h": 76.8,
    "lag_24h": 84.0, "lag_48h": 88.3, "lag_168h": 91.7,
    "rolling_mean_24h": 81.2, "rolling_std_24h": 6.4,
    "rolling_mean_168h": 85.9, "rolling_std_168h": 11.2
  }]}'
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
| `make rebuild` | Rebuild all images and restart the stack |
| `make rebuild-api` | Rebuild only the API image |
| `make rebuild-mlflow` | Rebuild only the MLflow image |
| `make rebuild-fresh` | Rebuild all images with `--no-cache` and restart |
| `make ingest` | Download raw data from ENTSO-E |
| `make featurise` | Engineer features |
| `make train` | Train model and register in MLflow |
| `make evaluate` | Evaluate model and generate plots |
| `make all` | Run full pipeline (ingest → featurise → train → evaluate) |
| `make test` | Run test suite |
| `make lint` | Run ruff + mypy |
| `make compare` | Print MLflow runs for comparison |
| `make clean` | Remove processed and reference data files |
| `make clean-raw` | Remove raw data + manifests (keeps `.gitkeep` placeholders) |

## Project Structure

```
.
├── data/
│   ├── raw/            # Raw CSVs + manifests (local; regenerated by make ingest)
│   ├── processed/      # features.parquet + features_schema.json (local)
│   └── reference/      # Reference data for drift detection (local)
├── docs/               # Documentation (local only; gitignored by design)
├── logs/               # Per-stage run logs (local)
├── models/             # Saved models (local)
├── notebooks/          # Exploration notebooks (local)
├── reports/            # Evaluation plots and reports (local)
├── secrets/            # API token, env, passwords (local; .env.example committed)
├── src/
│   ├── cli.py          # `python -m src <stage>` dispatch
│   ├── common/         # Shared utilities: metrics, splits, schema
│   ├── config/         # Config loading and validation
│   ├── evaluation/     # Stage 4: metrics, plots, reporting
│   ├── features/       # Stage 2: calendar, lags, availability, weather merge
│   ├── ingestion/      # Stage 1: ENTSO-E, validation, generation, weather cache
│   ├── monitoring/     # Drift monitoring
│   ├── serving/        # Stage 5: FastAPI app + request validation
│   └── training/       # Stage 3: loader, MLflow registry
├── tests/              # Test suite (mirrors src/ packages)
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── functional/     # End-to-end tests
├── params.yaml         # Pipeline configuration
├── pyproject.toml      # Project metadata and dependencies (single source of truth)
├── Dockerfile          # API image
├── Dockerfile.mlflow   # MLflow tracking-server image
├── docker-compose.yml  # Postgres + MLflow + API services
└── Makefile            # Build automation
```

## What's Committed vs. Local

The repository deliberately tracks **source** and **templates**, and leaves
**generated output** on your machine (or in MLflow). The rule is:

> Commit what you **author**; ignore what you **generate**.
> Provenance travels with its artifact, in the store of record.

| Class | Committed? | Where it lives |
|-------|-----------|----------------|
| Authored source (`src/`, `tests/`, `.github/`, `params.yaml`, `pyproject.toml`, `Makefile`, `Dockerfile*`, `docker-compose.yml`, `README.md`) | ✅ yes | GitHub |
| Bootstrap templates (`secrets/.env.example`, `.gitkeep` placeholders) | ✅ yes | GitHub |
| Pipeline output (`data/**` incl. **every `manifest.json`**, `models/`, `reports/`, `logs/`) | ❌ no | local disk |
| MLflow runs, artifacts and registry (`mlruns/`, `mlartifacts/`) | ❌ no | local Docker volume |
| Secrets (`secrets/.env`, `secrets/*.txt`) | ❌ no | local disk |
| Caches and build artifacts (`__pycache__/`, `*_cache/`, `.venv/`, `build/`, `dist/`, `*.egg-info/`, `backups/`) | ❌ no | local disk |
| Docs and notebooks (`docs/`, `notebooks/` scratch) | ❌ no | local disk |

**Why aren't the data manifests committed?** A manifest is *generated* by
`make ingest` (it records `downloaded_at`, so it changes on every run) and its
only consumer is the model it describes. `src/training/registry.py` logs
`manifest.json` into the model's own MLflow run as a `config/` artifact and
tags that run with `manifest_sha256`, so any model can always be traced back to
the exact raw-data snapshot it was trained on — without a single file in git.
Since `models/` and `mlruns/` are local, the manifest that describes them is
local for exactly the same reason; committing it would only add churn.

Everything ignored can be regenerated from `params.yaml`:
`make ingest` → `make featurise` → `make train` → `make evaluate`
(or simply `make all`). The full policy lives as a header comment in
`.gitignore`.

## License

MIT