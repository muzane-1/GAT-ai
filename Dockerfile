# GAT-ai — slim CPU-ready CI/training container.
#
# The image doubles as the CI executor: every workflow step (ruff, mypy,
# bandit, pytest --cov, nbconvert, 1-epoch train, Optuna smoke) runs inside
# `docker run --rm aml-gnn:ci <cmd>`. Keep the base minimal (no CUDA) so the
# wheel downloads stay small and the layer cache stays warm.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies in their own layer so source edits reuse the cache.
COPY requirements.txt ./
# Pin an explicit CPU torch wheel first; `torch>=2.2,<2.8` in requirements.txt
# is then already satisfied and pip skips the multi-GB PyPI wheel.
RUN pip install "torch>=2.2,<2.8" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Run as an unprivileged user. /app is owned by the app user so runtime
# artifacts (checkpoints/, metrics history, pytest/ruff/mypy caches) stay
# writable during CI runs.
RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /home/appuser --shell /bin/bash appuser \
    && mkdir -p /app/checkpoints /app/data/processed \
    && chown -R appuser:appuser /app
USER appuser

# Default entrypoint: train the GATv2 AML detector. Override with any CI step,
# e.g. `docker run --rm aml-gnn:ci pytest tests/` or `ruff check .`.
CMD ["python", "-m", "src.training.train"]
