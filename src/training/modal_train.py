"""Modal.ai entry point for training on the persisted transaction graph."""

from __future__ import annotations

import argparse
from typing import Any

from src.storage.pipeline import load_parquet

try:
    import modal
except ImportError:  # Modal is only needed when deploying this module.
    modal = None  # type: ignore[assignment]


if modal is not None:
    image = modal.Image.debian_slim().pip_install(
        "torch", "torch-geometric", "duckdb", "playwright"
    )
    volume = modal.Volume.from_name("project-data-vol", create_if_missing=True)
    app = modal.App("aml-gnn-training")

    @app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
    def train_remote(
        parquet_path: str = "/data/transactions.parquet",
        learning_rate: float = 0.005,
        batch_size: int = 2000,
        epochs: int = 10,
    ) -> dict[str, Any]:
        """Train from a mounted Parquet file and persist the best checkpoint."""
        del batch_size  # Existing trainer is transductive/full-graph.
        from src.data_pipeline.graph_builder import build_pyg_data
        from src.training.train import train_model
        from src.utils.config import load_config

        config = load_config()
        config["training"].update({"lr": learning_rate, "epochs": epochs})
        frame = load_parquet(parquet_path)
        local_csv = "/tmp/transactions.csv"
        frame.to_csv(local_csv, index=False)
        config["data"]["raw_source"] = local_csv
        config["paths"].update(
            {
                "checkpoints_dir": "/data/checkpoints",
                "metrics_history": "/data/checkpoints/metrics_history.json",
                "best_checkpoint": "/data/checkpoints/best.pt",
            }
        )
        build_pyg_data(frame)  # Validate the mounted dataset before training.
        return train_model(config, epochs=epochs)
else:
    image = volume = app = None

    def train_remote(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Report a useful error when Modal is not installed locally."""
        raise RuntimeError("Modal is required to run train_remote; install 'modal'.")


def main() -> None:
    """Run the deployed function through ``modal run``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if modal is None:
        raise RuntimeError("Modal is required; install 'modal' before running this command")
    train_remote.remote(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
