"""Hyperparameter optimisation via Optuna.

Optimises validation PR-AUC (the AML-friendly choice for heavy imbalance)
over a joint search space of learning rate, hidden dimension, attention
heads, dropout, and Focal Loss alpha/gamma. Every trial clones the config,
runs a shortened training loop, and reports the best validation metric.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from src.training.train import train_model
from src.utils import get_logger, load_config

logger = get_logger(__name__)


def sample_config(trial: optuna.Trial, base_config: dict[str, Any]) -> dict[str, Any]:
    """Build a per-trial config by sampling model/loss/training parameters.

    Args:
        trial: Active Optuna trial.
        base_config: Baseline YAML configuration.

    Returns:
        Configuration clone with sampled hyperparameters applied.
    """
    config = json.loads(json.dumps(base_config))  # deep copy via JSON round-trip

    config["model"]["hidden_channels"] = trial.suggest_int("hidden_channels", 32, 128, step=16)
    heads_choices = sorted({h for h in range(1, 9) if config["model"]["hidden_channels"] % h == 0})
    config["model"]["heads"] = trial.suggest_categorical("heads", heads_choices)
    config["model"]["dropout"] = trial.suggest_float("dropout", 0.1, 0.6)

    config["loss"]["init_alpha"] = trial.suggest_float("init_alpha", 0.1, 0.7)
    config["loss"]["init_gamma"] = trial.suggest_float("init_gamma", 1.0, 4.0)

    config["training"]["lr"] = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    config["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    return config


def objective(trial: optuna.Trial, base_config: dict[str, Any], epochs: int) -> float:
    """Single Optuna objective: shortened training + validation PR-AUC."""
    config = sample_config(trial, base_config)
    run_dir = Path("checkpoints") / "optuna" / f"trial-{trial.number}"
    summary = train_model(config, epochs=epochs, run_dir=run_dir, quiet=True)

    pr_auc = summary["best_val_metrics"].get("pr_auc", float("nan"))
    if np.isnan(pr_auc):
        return 0.0
    return float(pr_auc)


def run_study(
    config: dict[str, Any], n_trials: int | None = None, epochs: int | None = None
) -> optuna.Study:
    """Execute the Optuna study.

    Args:
        config: Baseline configuration.
        n_trials: Optional trial-count override.
        epochs: Optional per-trial epoch override.

    Returns:
        The finished Optuna study.
    """
    tuning_cfg = config["tuning"]
    trials = int(n_trials or tuning_cfg["n_trials"])
    per_trial_epochs = int(epochs or tuning_cfg["epochs_per_trial"])

    storage = tuning_cfg.get("storage") or None
    study = optuna.create_study(
        study_name=tuning_cfg.get("study_name", "aml-gatv2"),
        direction="maximize",
        storage=storage,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        load_if_exists=bool(storage),
    )
    study.optimize(
        lambda trial: objective(trial, config, per_trial_epochs),
        n_trials=trials,
        timeout=tuning_cfg.get("timeout_seconds"),
    )

    best = study.best_trial
    logger.info("Best trial #%d PR-AUC=%.4f params=%s", best.number, best.value, best.params)

    out_dir = Path(config["paths"]["checkpoints_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_params.json").write_text(
        json.dumps({"value": best.value, "params": best.params}, indent=2)
    )
    return study


def main() -> None:
    """CLI entrypoint for hyperparameter optimisation."""
    parser = argparse.ArgumentParser(description="Tune the AML GATv2 model with Optuna")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config path")
    parser.add_argument("--trials", type=int, default=None, help="Trial-count override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per trial override")
    args = parser.parse_args()

    config = load_config(args.config)
    run_study(config, n_trials=args.trials, epochs=args.epochs)


if __name__ == "__main__":
    main()
