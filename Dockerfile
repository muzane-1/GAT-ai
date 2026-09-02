# GAT-ai — slim CPU-ready CI/training container.
#
# The image doubles as the CI executor: every workflow step (ruff, mypy,
# bandit, pytest --cov, nbconvert, 1-epoch train, Optuna smoke) runs inside
# `docker run --rm aml-gnn:ci <cmd>`. Keep the base minimal (no CUDA) so the
# wheel downloads stay small and the layer cache stays warm.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_RETRIES=10 \
    PIP_DEFAULT_TIMEOUT=120

# Install dependencies in their own layers so source edits reuse the cache.
# Layer 1: install the small SymPy prerequisite separately so the CPU Torch
# wheel layer remains cacheable.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "mpmath>=1.3,<2"
# Layer 2: pin an explicit CPU torch wheel first; `torch>=2.2,<2.8` in
# requirements.txt is then already satisfied and pip skips the multi-GB
# PyPI wheel. mpmath (sympy dependency) is already installed from layer 1.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==2.7.1+cpu" --index-url https://download.pytorch.org/whl/cpu
# NeighborLoader requires a compiled sampling backend. Install the CPU wheel
# matching the supported Torch 2.7 series used by the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyg_lib \
    -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
# Layer 3: everything else.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Run as an unprivileged user. /app is owned by the app user so runtime
# artifacts (checkpoints/, metrics history, pytest/ruff/mypy caches) stay
# writable during CI runs.
RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /home/appuser --shell /bin/bash appuser \
    && mkdir -p /home/appuser /app/checkpoints /app/data/processed \
    && chown -R appuser:appuser /home/appuser /app \
    && chmod 700 /home/appuser
USER appuser

# Default entrypoint: train the GATv2 AML detector. Override with any CI step,
# e.g. `docker run --rm aml-gnn:ci pytest tests/` or `ruff check .`.
CMD ["python", "-m", "src.training.train"]
