# =============================================================================
# MLOps Energy Forecast — Makefile
# =============================================================================
# Single-command workflow for the full pipeline.
#
# Prerequisites:
#   - GNU Make installed (Linux/macOS: built-in; Windows: install via
#     Chocolatey `choco install make` or Scoop `scoop install make`)
#   - Docker Compose running (run `make up` first)
#   - Python .venv activated for local commands
#
# NOTE: modules are run via `python -m src.<module>` so that the `src`
# package (and its shared helpers in src/utils.py) are importable.
# =============================================================================

.PHONY: ingest featurise train evaluate serve test compare all up down clean

# --- Pipeline steps (run locally, need MLflow running in Docker) ---

ingest:
	python -m src.ingest

featurise:
	python -m src.featurise

train:
	python -m src.train

evaluate:
	python -m src.evaluate

# --- Docker management ---

up:
	docker compose --env-file secrets/.env up -d

down:
	docker compose down

serve:
	docker compose --env-file secrets/.env up -d api

# --- Model comparison ---

compare:
	@echo "Opening MLflow UI to compare runs..."
	@echo "Navigate to: http://localhost:5000"
	@docker compose exec mlflow mlflow models list || true

# --- Testing ---

test:
	pytest tests/ -v

# --- Full pipeline ---

all: ingest featurise train evaluate
	@echo "=== Pipeline complete ==="

# --- Cleanup ---

clean:
	rm -rf data/processed/*.parquet data/reference/*.parquet
	@echo "Cleaned processed and reference data."