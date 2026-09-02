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

### Start MLflow (Docker)

```bash
make up
```

### Run Pipeline Locally

```bash
make ingest      # Download raw data
make featurise   # Engineer features
make train       # Train model
make evaluate    # Evaluate model
```

Or run the full pipeline:

```bash
make all
```

### Run Tests

```bash
make test
# or directly:
pytest tests/ -v
```

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