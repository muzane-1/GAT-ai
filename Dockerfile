FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# PyTorch CPU build — keeps the image slim and CI-friendly.
RUN pip install "torch>=2.2,<2.8" --index-url https://download.pytorch.org/whl/cpu \
    && pip install torch_geometric pandas numpy scikit-learn PyYAML requests optuna pytest ruff

COPY . /app

CMD ["python", "-m", "src.training.train"]
