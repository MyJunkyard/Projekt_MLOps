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
# NOTE: pipeline stages are invoked via the single CLI entry point
# `python -m src <command>` (src/cli.py), so the `src` package and its
# role-based subpackages (common/, ingestion/, features/, training/,
# evaluation/) are importable. Stage 5's Airflow DAG adopts the same
# commands.
# =============================================================================

.PHONY: ingest featurise train evaluate serve test lint compare all up down rebuild rebuild-api rebuild-mlflow rebuild-fresh clean clean-raw check-env

# --- Pipeline steps (run locally, need MLflow running in Docker) ---

# Preflight: fail `make ingest` early with an actionable message when no
# ENTSO-E key source is configured and synthetic fallback is not opted in.
# Local runs use ENTSOE_API_KEY; Docker uses ENTSOE_API_TOKEN_FILE — exactly
# one source (see _read_api_key in src/ingestion/entsoe.py). Never auto-load
# secrets/entsoe_api_token.txt locally: that would break the single-source
# invariant and risk leaking the key into child processes.
check-env:
ifndef ENTSOE_API_KEY
	ifndef ENTSOE_API_TOKEN_FILE
		@echo "ENTSO-E API key not configured."
		@echo "Local: set ENTSOE_API_KEY (PowerShell: \$$env:ENTSOE_API_KEY = (Get-Content secrets/entsoe_api_token.txt -Raw).Trim())"
		@echo "Docker: mount the token as ENTSOE_API_TOKEN_FILE (see docker-compose.yml)."
		@echo "Or set data.entsoe.allow_synthetic: true explicitly for offline dev/CI runs only."
		@exit 1
	endif
endif

ingest: check-env
	python -m src ingest

featurise:
	python -m src featurise

train:
	python -m src train

evaluate:
	python -m src evaluate

# --- Docker management ---

up:
	docker compose --env-file secrets/.env up -d

down:
	docker compose down

serve:
	docker compose --env-file secrets/.env up -d api

rebuild:
	docker compose --env-file secrets/.env up -d --build

rebuild-api:
	docker compose --env-file secrets/.env up -d --build api

rebuild-mlflow:
	docker compose --env-file secrets/.env up -d --build mlflow

rebuild-fresh:
	docker compose --env-file secrets/.env build --no-cache
	docker compose --env-file secrets/.env up -d

# --- Model comparison ---

compare:
	@echo "Opening MLflow UI to compare runs..."
	@echo "Navigate to: http://localhost:5000"
	@docker compose exec mlflow mlflow models list || true

# --- Testing and static analysis ---

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy

# --- Full pipeline ---

all: ingest featurise train evaluate
	@echo "=== Pipeline complete ==="

# --- Cleanup ---

clean:
	rm -rf data/processed/*.parquet data/reference/*.parquet
	@echo "Cleaned processed and reference data."

# Raw data AND its manifests are local (gitignored). This removes them so the
# next `make ingest` performs a clean download; .gitkeep placeholders and the
# empty cache directories are preserved.
clean-raw:
	find data/raw -type f -name '*.csv' -delete
	find data/raw -type f -name 'manifest.json' -delete
	@echo "Cleaned raw data and manifests (kept .gitkeep placeholders)."