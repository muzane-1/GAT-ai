# Agent Notes: AML GNN Repository

## Quick commands
- Install: `pip install "torch>=2.2,<2.8" --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
- Tests: `python -m pytest tests/`
- Lint: `python -m ruff check . && python -m ruff format --check .`
- Train: `python -m src.training.train (--epochs N)`
- Tune: `python -m src.training.tune (--trials N --epochs N)`
- Drift monitor: `python scripts/update_pipeline.py`
- Docker: `sudo docker build -t aml-gnn:ci .` (dockerd already runs; daemon socket requires sudo)

## Conventions
- The data plane lives in `src/pipeline/` with three MLOps layers: `discovery/` (auto_fetch, scraper, agents), `validation/` (pandera_schema, quality scoring, substructure_verifier), `orchestration/` (step-based fetch -> validate -> transform). Graph kernel: `src/pipeline/transform/`.
- Central config: `config/config.yaml` — everything (model/loss/training/data) reads from it.
- Canonical transaction schema: `tx_id, src, dst, amount, timestamp, is_laundering`. Aliases (incl. `from_address`/`to_address`) are normalised in `src/pipeline/discovery/ingestion.py::normalize_columns`.
- Node features live in `src/pipeline/transform/features.py::FEATURE_COLUMNS` (9 columns). Edge features = [log1p(amount), normalised timestamp delta].
- Fall back to synthetic data when a data source fails: keep checker-friendly; synthetic rings mix *only* among flagged accounts so labels stay clean.
- Checkpoints & metrics history land in `checkpoints/` (gitignored) — `scripts/update_pipeline.py` uses it for drift decisions.
- ruff: line-length 100, rule set E/F/I/UP/B/SIM/ANN (ANN401 ignored). Format on save.

## Validation status (last verified)
- 22 unit tests pass; ruff clean; 1-epoch training + Optuna smoke succeed inside Docker.
