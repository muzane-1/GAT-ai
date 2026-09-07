# GAT-ai — Financial AML Graph AI Pipeline

Production-grade **Anti-Money Laundering (AML) detection** built as an
end-to-end MLOps pipeline: multi-provider *data discovery* → Pandera-validated
*ingestion & quality gates* → **PyTorch Geometric (PyG)** graph transformation →
**GATv2** attention training with an adaptive Focal Loss for extreme class
imbalance.

The data plane is organised into **three explicit MLOps operational layers**
under `src/pipeline/`:

| Layer | Package | Responsibility |
|---|---|---|
| **1. Discovery & Ingestion** | `src/pipeline/discovery/` | Agentic dataset discovery (Kaggle / GitHub / Hugging Face / keyless web), bounded async Playwright scraping with CAPTCHA-challenge surfacing, CSV/HF fetching, deterministic synthetic fallback, LangGraph/CrewAI-ready agent nodes |
| **2. Validation & Quality** | `src/pipeline/validation/` | **Pandera** schema contract (`src`/`dst`/`from_address`/`to_address`, `amount`, `timestamp`, `is_laundering`), dataset-quality scoring (schema / health / topology), PyG **substructure verifier** (directed cycles, smurfing fans, k-core) |
| **3. Orchestration & Pipeline** | `src/pipeline/orchestration/` | Step-based `fetch → validate → transform` execution — ZenML / Kedro compatible `@step` wrappers with declared artifact inputs/outputs |

The shared **graph transformation kernel** (`src/pipeline/transform/`) converts
validated transaction tables into PyG `Data` objects with topology features and
positional encodings.

## Directory Structure

```
GAT-ai/
├── config/
│   └── config.yaml              # Single source of truth: model, loss, training, data, monitoring
├── data/                        # (gitignored) raw / discovery / processed artifacts
├── scripts/
│   ├── update_pipeline.py       # Checkpoint registry + metric drift + retrain trigger
│   └── verify_readiness.py      # Deterministic end-to-end readiness check
├── src/
│   ├── pipeline/                # ── THE DATA PLANE (3 MLOps layers) ──
│   │   ├── discovery/           #   Layer 1: auto_fetch, scraper (Playwright), ingestion, agents
│   │   ├── validation/          #   Layer 2: pandera_schema, quality scoring, substructure_verifier
│   │   ├── transform/           #   Kernel:  features, graph_builder, positional_encoding, sampling
│   │   ├── orchestration/       #   Layer 3: steps.py (@step), pipeline.py (fetch → validate → transform)
│   │   └── __init__.py          #   Facade re-exporting the full public API
│   ├── models/                  # GATv2Net, GATv2GraphTransformer, AdaptiveFocalLoss
│   ├── training/                # train.py, tune.py (Optuna), modal_train.py (GPU)
│   ├── storage/                 # JSONL cleaning + Parquet/PyG persistence (Modal Volume backed)
│   ├── utils/                   # config loader, logger, metrics, legacy shims
│   ├── dataset.py               # Legacy PyG InMemoryDataset loader (HF fallback)
│   └── model.py                 # Legacy shim re-exporting GATv2 / GATv2AMLModel
├── tests/                       # pytest suite (pipeline, agents, pandera, orchestration, models, training)
├── notebooks/                   # EDA, prototyping, attention visualisation, pipeline validation
├── train.py                     # Thin shim delegating to src.training.train
├── Dockerfile                   # Slim CPU container doubling as the CI executor
└── .github/workflows/ci.yml     # ruff → mypy → bandit → pytest --cov → nbconvert → 1-epoch → Optuna smoke
```

## MLOps Tool Stack

| Concern | Tools |
|---|---|
| **Discovery** | Playwright (async scraping + CAPTCHA audio-solver hooks), LangGraph / CrewAI (agent node scaffolding in `discovery/agents.py`), `huggingface_hub`, Kaggle & GitHub Search APIs |
| **Validation** | **Pandera** (`validation/pandera_schema.py`), PyG **Substructure Verifier** (`validation/substructure_verifier.py`), scipy connectivity checks |
| **Orchestration** | ZenML / Kedro-compatible step wrappers (`orchestration/steps.py`), Optuna (HPO), Modal (remote GPU), GitHub Actions + Docker (CI) |
| **ML** | PyTorch Geometric (PyG) — GATv2 attention, NeighborLoader sampling, Laplacian / random-walk positional encodings, adaptive Focal Loss |

## The 3-Layer Pipeline

```python
from src.pipeline import run_pipeline

# One call executes fetch → validate → transform.
# Without a source it deterministically falls back to the synthetic generator.
catalog = run_pipeline()
graph = catalog["graph"]  # torch_geometric.data.Data
report = catalog["validation_report"]  # rows / nodes / aml_ratio / pandera gate
```

### Layer 1 — Discovery & Ingestion (`src/pipeline/discovery/`)

```python
from src.pipeline.discovery import (
    discover_and_verify,
    fetch_transactions,
    verified_summary,
    PlaywrightScraper,
    ScraperConfig,
    run_discovery_agents,
)

# Agentic discovery: plan queries → fan out to providers → verify raw bytes.
verified = discover_and_verify(providers="kaggle,github,web", top_k=3, offline=False)
summary = verified_summary(verified)  # quality_score, has_explicit_label, local_path, ...

# LangGraph / CrewAI-ready: the same logic as plan → discover → verify nodes.
state = run_discovery_agents({"offline": True, "assets": ["btc"], "aml_terms": ["aml"]})
# build_discovery_graph() returns a compiled langgraph StateGraph when
# `langgraph` is installed, else a dependency-free sequential executor.
```

`scraper.py` captures JSON responses from JavaScript-rendered pages with
storage-state reuse. CAPTCHAs are never bypassed silently: the challenge is
surfaced to a caller-supplied `audio_solver` (audio CAPTCHA solving) or
`captcha_token`. Every remote path fails safely into the deterministic
synthetic generator, so CI stays green without credentials.

### Layer 2 — Validation & Quality (`src/pipeline/validation/`)

```python
from src.pipeline.validation import evaluate_candidate_dataset, validate_transaction_schema

validated = validate_transaction_schema(
    raw_df
)  # Pandera gate (aliases incl. from_address/to_address)
scores = evaluate_candidate_dataset(raw_df)  # schema_fit / health / topology / weighted_score
```

`substructure_verifier.py` cross-checks GNN predictions against local AML
motifs (directed cycles, smurfing fans, dense k-core subgraphs) and fires an
independent retrain trigger from `scripts/update_pipeline.py`.

### Layer 3 — Orchestration (`src/pipeline/orchestration/`)

```python
from src.pipeline.orchestration import build_default_pipeline, validate_step

pipeline = build_default_pipeline()  # [fetch_step, validate_step, transform_step]
catalog = pipeline.run(source="data/raw/transactions.csv")

# Every step is a ZenML / Kedro compatible unit with declared artifacts:
validate_step.inputs  # ("raw_df",)               <- threaded through the catalog
validate_step.outputs  # ("validated_df", "validation_report")
```

## Quickstart

```bash
# 1. Install (CPU)
pip install "torch>=2.2,<2.8" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Tests (full suite)
python -m pytest tests/

# 3. Run the full data pipeline (fetch → validate → transform)
python -c "from src.pipeline import run_pipeline; print(run_pipeline()['validation_report'])"

# 4. Fetch live datasets (discovery layer; keyless providers, safe fallback)
python -c "from src.pipeline.discovery import discover_and_verify, verified_summary; \
print(verified_summary(discover_and_verify(top_k=3)))"

# 5. Train / tune / monitor
python -m src.training.train --epochs 100
python -m src.training.tune --trials 50 --epochs 100
python scripts/update_pipeline.py  # drift monitor + retrain trigger
```

## Model & Training

`src/models/` implements a **GATv2** attention network (plus a hybrid graph
transformer variant) with edge-feature injection and an **AdaptiveFocalLoss**
whose α/γ adapt to the observed FN/FP trade-off. Node features (9 behavioural
+ topological columns) are standard-scaled; edge features carry the scaled
amount and a normalised timestamp delta. Configuration lives exclusively in
`config/config.yaml`.

## Legacy compatibility

Notebooks written against the previous layout keep working: `src.dataset`,
`src.model`, and `src.utils` remain importable with the same legacy symbols,
and the `src.pipeline` facade re-exports the entire former
`src.data_pipeline` / `src.ingestion` / `src.eval` public API.

## CI

The GitHub Actions workflow builds the Docker image and runs `ruff` (lint +
format), `mypy`, `bandit`, `pytest --cov` (≥80%), headless notebook execution,
a 1-epoch training dry-run, and an Optuna smoke trial — all inside the
container.
