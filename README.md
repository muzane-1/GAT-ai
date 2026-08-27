# GAT-ai — Graph Neural Network AML Detection

Production-grade PyTorch Geometric (PyG) Anti-Money Laundering detection built
around a **GATv2** attention architecture with an adaptive Focal Loss to
handle extreme class imbalance.

## Repository layout

```
config/config.yaml          # All parameters (model, loss, training, tuning, monitoring)
src/
  dataset.py                # Legacy PyG InMemoryDataset loader (HF fallback)
  model.py                  # Legacy shim re-exporting GATv2 / GATv2AMLModel
  utils/legacy.py           # Legacy FocalLoss, checkpoint helpers, compute_metrics
  data_pipeline/            # ingestion / features / graph_builder
  models/                   # GATv2Net and AdaptiveFocalLoss
  training/                 # train.py (early stop, metrics) and tune.py (Optuna)
  utils/                    # logger, metrics, config loader
scripts/update_pipeline.py  # Checkpoint registry + metric drift + retrain trigger
tests/                      # Unit tests for pipeline, models, training
Dockerfile                  # Slim CPU-ready container
.github/workflows/          # Lint -> tests -> 1-epoch dry-run -> Optuna smoke
notebooks/                  # Historical exploration notebooks (kept working)
train.py                    # Thin shim delegating to src.training.train
```

## Quick start

```bash
pip install -r requirements.txt
python train.py --epochs 100
python -m src.training.tune --trials 50 --epochs 100
python scripts/update_pipeline.py
```

## Legacy compatibility

Notebooks written against the previous layout keep working: `src.dataset`,
`src.model`, and `src.utils` remain importable with the same legacy symbols
(`GATv2`, `GATv2AMLModel`, `FocalLoss`, `compute_metrics`, checkpoint helpers).
The root `train.py` is preserved as a shim.

## CI

The GitHub Actions workflow builds the Docker image, runs `ruff`, executes
`pytest`, a 1-epoch training dry-run, and an Optuna smoke trial — all inside
the container.