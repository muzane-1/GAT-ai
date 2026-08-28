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
  data_pipeline/            # ingestion / auto_fetch / features / graph_builder
  models/                   # GATv2Net and AdaptiveFocalLoss
  training/                 # train.py (early stop, metrics) and tune.py (Optuna)
  utils/                    # logger, metrics, config loader
scripts/update_pipeline.py  # Checkpoint registry + metric drift + retrain trigger
tests/                      # Unit tests for pipeline, models, training
Dockerfile                  # Slim CPU-ready container
.github/workflows/          # Lint -> tests -> 1-epoch dry-run -> Optuna smoke
notebooks/                  # Interactive sandbox + validation notebooks
train.py                    # Thin shim delegating to src.training.train
```

## Automated Data Pipeline (API Ingestion)

`src/data_pipeline/auto_fetch.py` orchestrates fetch → sanitize → validate as
three composable stages, so the training pipeline never hard-crashes on a
missing or malformed remote dataset:

1. **Discovery & fetch** — `list_candidate_datasets()` queries the Hugging
   Face Hub (`HfApi.list_datasets`) for AML-flavoured transaction datasets;
   `auto_fetch()` then resolves a source in priority order — explicit local
   path / URL → HF dataset id (default `qubit420/ibm-aml-LI-smaller`) →
   deterministic synthetic generator.
2. **Sanitation** — `sanitize_transactions()` normalises column aliases into
   the canonical schema (`tx_id, src, dst, amount, timestamp,
   is_laundering`), parses/coerces types, clips negative amounts, drops
   duplicate edges, and fills missing ids/labels/timestamps. `auto_fetch()`
   optionally applies a `log1p` amount normalisation.
3. **Validation** — `validate_transactions()` computes node/edge counts,
   undirected connectivity (scipy), null-cell count, and the AML class
   ratio; `fetch_to_pyg()` converts the verified table into a PyG `Data`
   object via the canonical `graph_builder`.

```python
from src.data_pipeline import auto_fetch, fetch_to_pyg

df, stats = auto_fetch(hf_query="qubit420/ibm-aml-LI-smaller")
data, graph_stats = fetch_to_pyg(hf_query="qubit420/ibm-aml-LI-smaller")
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Graph topology + class-distribution EDA |
| `02_model_prototyping.ipynb` | Dry-run forward pass + short training loop |
| `03_visualize_attention.ipynb` | Attention weights + embedding projection |
| `04_data_pipeline_validation.ipynb` | Auto-fetch sanitation & validation checks |

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