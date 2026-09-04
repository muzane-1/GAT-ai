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
  eval/                     # Dataset evaluation engine (schema / health / topology / scoring)
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

`src/data_pipeline/auto_fetch.py` orchestrates **agentic discovery** → fetch → sanitize → validate as
four composable stages, so the training pipeline never hard-crashes on a
missing or malformed remote dataset:

1. **Agentic discovery & fetch** — `generate_search_queries()` plans a
   deterministic cross-product of crypto assets × AML/graph terms × output
   formats; `discover_candidates()` fans the queries out to Kaggle, GitHub,
   Hugging Face and a keyless web search; `discover_and_verify()` downloads
   the top-K candidates and re-scores them on the raw bytes. `auto_fetch()`
   then resolves a source in priority order — explicit local path / URL →
   HF dataset id → dynamically discovered candidates → deterministic synthetic
   generator.
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

## Agentic Dataset Discovery Pipeline

`src/data_pipeline/auto_fetch.py` also exposes a standalone, fully-typed
agentic discovery layer that searches public sources and returns **verified**
dataset metadata + raw download paths:

- **Query planner** — `generate_search_queries()` deterministically builds
  `[Crypto Asset (BTC, ETH, Solana)] + [AML/Graph terms (transaction graph,
  illicit, money laundering, fraud)] + [Formats (CSV, Parquet, PyG, NetworkX)]`
  combinations; an LLM-backed planner can swap in behind the same signature.

- **Providers** — `discover_candidates()` fans queries out to:
  - **Kaggle** (official `kaggle` client — `pip install kaggle` — requiring
    `KAGGLE_USERNAME`/`KAGGLE_KEY`),
  - **GitHub** Search API (optionally authenticated with `GITHUB_TOKEN`),
  - **Hugging Face** Hub (always queried, via `HfApi.list_datasets`),
  - **Web** (keyless DuckDuckGo HTML endpoint, best-effort).
  Each provider call is guarded (network failures degrade gracefully) and
  cached for 300s; results are de-duplicated by id and ranked by a metadata
  heuristic.

- **Reliability gates (strict)** — `assess_reliability()` scores every candidate
  0-100 (:data:`MIN_QUALITY_SCORE = 60`) using the repository's eval engine
  (schema fit / data health / graph topology / class balance) plus:
    - `has_explicit_label` — the table must expose an explicit target label
      (`label`, `is_illicit`, `is_laundering`, `is_fraud`, ...), and
    - `has_edge_connections` — both `source`/`target`-style edge columns must be
      present. Candidates failing either gate are **hard-rejected**, never handed
      downstream.



- **Verified handoff** — `discover_and_verify(top_k=3)` downloads the top-K
  raw files into `data/discovery/`, re-scores them on the actual bytes and returns
  `VerifiedDataset` records; `verified_summary()` flattens them into JSON-safe
  metadata (local paths included). The pipeline-integration helpers
  pass raw materials straight into the existing modules:
  `handoff_to_ingestion()` → `src.data_pipeline.ingestion` (schema normalisation),
  `handoff_to_graph_builder()` → `src.data_pipeline.graph_builder` (PyG `Data`),
  `handoff_to_features()` → `src.data_pipeline.features` (scaling/class imbalance/
  feature mapping), and `handoff()` runs all three in sequence.

```python
from src.data_pipeline import discover_and_verify, verified_summary

verified = discover_and_verify(  # offline=True in CI/key-less environments
    providers="kaggle,github,web",  # or AML_DISCOVERY_PROVIDERS env var
    top_k=3,
    offline=False,
)
summary = verified_summary(verified)
# [{'id': 'github:acme/aml-graphs', 'quality_score': 92.0,
#   'has_explicit_label': True, 'has_edge_connections': True,
#   'local_path': 'data/discovery/aml-graphs.csv', ...}, ...]
```

All discovery calls fail safely: when no internet, no API keys, or every
candidate fails the gates, the pipeline falls back to the deterministic synthetic
generator (`--offline` / no credentials),so CI (`verify_readiness.py`) stays
deterministic and green.

## Dataset Evaluation Engine (`src.eval`)

The dataset-quality logic extracted in `src/eval/` scores **raw, unsanitised**
candidate tables *before* the ingestion pipeline makes its hard validation
checks — this is how `auto_fetch()` ranks multiple Hugging Face candidates and
picks the best one. Each module is a single, testable concern:

| Module | Evaluator | What it measures |
|---|---|---|
| `src/eval/schema.py` | `evaluate_schema_fit()` | How much of the canonical schema (`tx_id, src, dst, amount, timestamp, is_laundering`) is present, mapping columns case-/whitespace-insensitively via `SCHEMA_ROLES` aliases (`source`, `target`, `value`, `label`, ...). Exposes `resolve_column()` and `CANONICAL_SCHEMA`. |
| `src/eval/health.py` | `evaluate_data_health()` | Non-null ratio, share of strictly positive amounts, and parseable-timestamp ratio (degrades gracefully to 0 on unparseable timestamps). |
| `src/eval/topology.py` | `evaluate_graph_topology()` | Node/edge counts, connectivity ratio (edges per node, capped at 1), AML class ratio and `aml_balance` — a useful class ratio is strictly between 0 and 0.5. |
| `src/eval/scoring.py` | `evaluate_candidate_dataset()` | Aggregates the above into a single `weighted_score` using `WEIGHTS = {schema_fit: 0.3, data_health: 0.3, graph_topology: 0.2, aml_balance: 0.2}`, plus raw `nodes`, `edges` and `aml_ratio` metrics. |

The return shape of `evaluate_candidate_dataset()` is backward compatible with
the original `src.data_pipeline.auto_fetch` implementation, so existing callers
(and tests) keep working.

```python
from src.eval import evaluate_candidate_dataset

scores = evaluate_candidate_dataset(df_raw)
# {'schema_fit': 1.0, 'data_health': 0.98, 'graph_topology': ...,
#  'aml_balance': 1.0, 'weighted_score': 0.99, 'nodes': 120, ...}
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