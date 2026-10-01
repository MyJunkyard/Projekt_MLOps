FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install production dependencies from pyproject.toml (single source of truth)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Copy application configuration
COPY params.yaml .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
