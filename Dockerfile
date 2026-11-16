FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf-cache DATA_DIR=/app/data EVALS_DIR=/app/evals
WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA wheels.
# Build with `--build-arg INSTALL_ML=false` for a slim image using EMBEDDING_BACKEND=hashing.
ARG INSTALL_ML=true
COPY pyproject.toml ./
COPY app ./app
RUN pip install ".[ui]" \
 && if [ "$INSTALL_ML" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install ".[ml]"; \
    fi

COPY evals ./evals
RUN useradd -m nexus && mkdir -p /app/data/uploads /app/data/chroma /app/evals/results \
 && chown -R nexus /app
USER nexus

EXPOSE 8000 8501
# Default: API. docker-compose overrides the command for the UI service.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
