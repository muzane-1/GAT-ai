"""Modal.ai entry point for training on the persisted transaction graph.

The Modal function mounts the ``project-data-vol`` Volume at ``/data`` and
trains directly on the sanitized artifacts written by ``src.storage.pipeline``:

* ``/data/transactions.pt``      — PyG ``{"data", "scaler"}`` artifact (preferred)
* ``/data/transactions.parquet`` — canonical transaction table (fallback)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from src.storage.pipeline import load_parquet

try:
    import modal
except ImportError:  # Modal is only needed when deploying this module.
    modal = None  # type: ignore[assignment]


DEFAULT_PYG_PATH = "/data/transactions.pt"
DEFAULT_PARQUET_PATH = "/data/transactions.parquet"


def load_training_inputs(
    pyg_path: str = DEFAULT_PYG_PATH,
    parquet_path: str = DEFAULT_PARQUET_PATH,
) -> tuple[str, Any]:
    """Resolve the sanitized training input produced by ``src.storage.pipeline``.

    Prefers the PyG artifact; falls back to the Parquet table (converted to the
    canonical schema on load). Raises ``FileNotFoundError`` when neither exists.
    """
    from src.storage.pipeline import load_pyg_dataset

    if Path(pyg_path).exists():
        data, scaler = load_pyg_dataset(pyg_path)
        return "pyg", (data, scaler)
    if Path(parquet_path).exists():
        return "parquet", load_parquet(parquet_path)
    raise FileNotFoundError(
        f"No training artifacts found at {pyg_path!r} or {parquet_path!r}; "
        "run src.pipeline.discovery.auto_fetch.handoff_to_storage first"
    )


if modal is not None:
    image = modal.Image.debian_slim().pip_install(
        "torch",
        "torch-geometric",
        "pandas",
        "numpy",
        "scikit-learn",
        "PyYAML",
        "requests",
        "duckdb",
        "pyarrow",
        "playwright",
    )
    volume = modal.Volume.from_name("project-data-vol", create_if_missing=True)
    app = modal.App("aml-gnn-training")

    @app.function(image=image, gpu="T4", volumes={"/data": volume}, timeout=3600)
    def train_remote(
        pyg_path: str = DEFAULT_PYG_PATH,
        parquet_path: str = DEFAULT_PARQUET_PATH,
        learning_rate: float = 0.005,
        batch_size: int = 2000,
        epochs: int = 10,
    ) -> dict[str, Any]:
        """Train from the mounted Volume artifacts and persist the checkpoint."""
        del batch_size  # Existing trainer is transductive/full-graph.
        from src.training.train import train_model
        from src.utils.config import load_config

        config = load_config()
        config["training"].update({"lr": learning_rate, "epochs": epochs})
        config["paths"].update(
            {
                "checkpoints_dir": "/data/checkpoints",
                "metrics_history": "/data/checkpoints/metrics_history.json",
                "best_checkpoint": "/data/checkpoints/best.pt",
            }
        )

        kind, payload = load_training_inputs(pyg_path, parquet_path)
        if kind == "pyg":
            data, scaler = payload
            return train_model(config, epochs=epochs, data=data, scaler=scaler)

        frame = payload
        # B108: never hardcode /tmp — resolve a platform-appropriate temp file.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="aml_transactions_", delete=False, encoding="utf-8"
        ) as handle:
            local_csv = handle.name
        try:
            frame.to_csv(local_csv, index=False)
            config["data"]["raw_source"] = local_csv
            return train_model(config, epochs=epochs)
        finally:
            Path(local_csv).unlink(missing_ok=True)
else:
    image = volume = app = None

    def train_remote(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Report a useful error when Modal is not installed locally."""
        raise RuntimeError("Modal is required to run train_remote; install 'modal'.")


def run_remote(
    learning_rate: float = 0.005,
    batch_size: int = 2000,
    epochs: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Invoke the deployed ``train_remote`` function (delegation entrypoint)."""
    if modal is None:
        raise RuntimeError("Modal is required to run train_remote; install 'modal'.")
    return train_remote.remote(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        **kwargs,
    )


def main() -> None:
    """Run the deployed function through ``modal run``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    summary = run_remote(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    print(summary)  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
