"""Unit tests for the Optuna hyperparameter-tuning module."""

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from src.training.tune import main, objective, run_study, sample_config
from src.utils import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


class _FakeTrial:
    """Minimal stand-in for :class:`optuna.Trial`."""

    def __init__(self, number: int = 0, fixed: dict[str, Any] | None = None) -> None:
        self.number = number
        self._fixed = fixed or {}

    def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> int:
        return int(self._fixed.get(name, low))

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        return self._fixed.get(name, choices[0])

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        return float(self._fixed.get(name, low))


class _FakeBest:
    """Stand-in for ``study.best_trial``."""

    number = 1
    value = 0.85
    params = {"lr": 0.001, "heads": 4}


def test_sample_config_suggests_hyperparameters() -> None:
    config = load_config(CONFIG_PATH)
    original = json.loads(json.dumps(config))
    trial = _FakeTrial(
        fixed={
            "hidden_channels": 64,
            "heads": 4,
            "dropout": 0.3,
            "init_alpha": 0.25,
            "init_gamma": 2.0,
            "lr": 0.005,
            "weight_decay": 1e-4,
        }
    )
    sampled = sample_config(trial, config)
    assert sampled["model"]["hidden_channels"] == 64
    assert sampled["model"]["heads"] == 4
    assert sampled["model"]["dropout"] == 0.3
    assert sampled["loss"]["init_alpha"] == 0.25
    assert sampled["loss"]["init_gamma"] == 2.0
    assert sampled["training"]["lr"] == 0.005
    assert sampled["training"]["weight_decay"] == pytest.approx(1e-4)
    # A deep copy is returned; the base config is never mutated.
    assert config == original
    assert sampled is not config


def test_objective_returns_pr_auc() -> None:
    config = load_config(CONFIG_PATH)
    summary = {"best_val_metrics": {"pr_auc": 0.83, "f1": 0.5}}
    with mock.patch("src.training.tune.train_model", return_value=summary) as train:
        value = objective(_FakeTrial(), config, epochs=1)
    assert value == pytest.approx(0.83)
    train.assert_called_once()
    assert train.call_args.kwargs["epochs"] == 1
    assert train.call_args.kwargs["quiet"] is True


def test_objective_nan_pr_auc_returns_zero() -> None:
    config = load_config(CONFIG_PATH)
    summary = {"best_val_metrics": {"pr_auc": float("nan")}}
    with mock.patch("src.training.tune.train_model", return_value=summary):
        assert objective(_FakeTrial(), config, epochs=1) == 0.0


def _fake_study() -> mock.Mock:
    """Return a Mock study whose optimize() actually invokes the objective."""

    study = mock.Mock()
    study.best_trial = _FakeBest()

    def _optimize(func: Any, n_trials: int, timeout: float | None = None) -> None:
        for i in range(int(n_trials)):
            func(_FakeTrial(number=i))

    study.optimize.side_effect = _optimize
    return study


def test_run_study_optimizes_and_writes_best_params(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config["paths"]["checkpoints_dir"] = str(tmp_path)
    config["tuning"]["n_trials"] = 3
    config["tuning"]["epochs_per_trial"] = 1

    study = _fake_study()
    summary = {"best_val_metrics": {"pr_auc": 0.85}}
    with (
        mock.patch("src.training.tune.train_model", return_value=summary),
        mock.patch("optuna.create_study", return_value=study) as create_study,
    ):
        result = run_study(config, n_trials=3, epochs=1)

    assert result is study
    create_study.assert_called_once()
    payload = json.loads((tmp_path / "best_params.json").read_text(encoding="utf-8"))
    assert payload["value"] == pytest.approx(0.85)
    assert payload["params"] == {"lr": 0.001, "heads": 4}


def test_run_study_uses_config_defaults(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config["paths"]["checkpoints_dir"] = str(tmp_path)
    config["tuning"]["n_trials"] = 2
    config["tuning"]["epochs_per_trial"] = 1
    config["tuning"].pop("storage", None)

    study = _fake_study()
    summary = {"best_val_metrics": {"pr_auc": 0.9}}
    with (
        mock.patch("src.training.tune.train_model", return_value=summary),
        mock.patch("optuna.create_study", return_value=study) as create_study,
    ):
        run_study(config)

    kwargs = create_study.call_args.kwargs
    assert kwargs["storage"] is None
    assert kwargs["load_if_exists"] is False
    assert kwargs["direction"] == "maximize"


def test_run_study_with_storage(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    config["paths"]["checkpoints_dir"] = str(tmp_path)
    config["tuning"]["n_trials"] = 1
    config["tuning"]["epochs_per_trial"] = 1
    config["tuning"]["storage"] = f"sqlite:///{tmp_path / 'optuna.db'}"

    study = _fake_study()
    summary = {"best_val_metrics": {"pr_auc": 0.9}}
    with (
        mock.patch("src.training.tune.train_model", return_value=summary),
        mock.patch("optuna.create_study", return_value=study) as create_study,
    ):
        run_study(config)

    kwargs = create_study.call_args.kwargs
    assert kwargs["load_if_exists"] is True
    assert str(kwargs["storage"]).startswith("sqlite:///")


def test_main_runs_from_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int | None] = {}

    def _fake_run(
        config: dict[str, Any],
        n_trials: int | None = None,
        epochs: int | None = None,
    ) -> None:
        captured["n_trials"] = n_trials
        captured["epochs"] = epochs

    monkeypatch.setattr("src.training.tune.run_study", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["tune", "--config", str(CONFIG_PATH), "--trials", "1", "--epochs", "1"],
    )
    main()
    assert captured == {"n_trials": 1, "epochs": 1}
