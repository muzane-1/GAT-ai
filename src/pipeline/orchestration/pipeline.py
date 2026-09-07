"""Executable AML pipeline: fetch → validate → transform.

:class:`Pipeline` threads declared artifacts between :class:`Step` objects
through a catalog mapping — the same contract Kedro's runner and ZenML's
pipeline executor use — so the default ``fetch → validate → transform``
pipeline can be lifted into either framework without rewriting steps.
"""

from __future__ import annotations

from typing import Any

from src.pipeline.orchestration.steps import Step, fetch_step, transform_step, validate_step
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Pipeline:
    """A linear, step-based pipeline with artifact threading."""

    def __init__(self, steps: list[Step], name: str = "aml_pipeline") -> None:
        self.name = name
        self.steps = list(steps)

    def run(self, catalog: dict[str, Any] | None = None, **context: Any) -> dict[str, Any]:
        """Execute the steps in order, threading artifacts through ``catalog``.

        Args:
            catalog: Optional pre-populated artifact store.
            **context: Keyword context (e.g. ``source``) forwarded to the
                entry step (the step that declares no inputs).

        Returns:
            The final catalog mapping artifact names to values.
        """
        store: dict[str, Any] = dict(catalog or {})
        for step_ in self.steps:
            args = [store[artifact] for artifact in step_.inputs]
            result = step_.function(*args, **context) if not step_.inputs else step_.function(*args)
            if len(step_.outputs) == 1:
                store[step_.outputs[0]] = result
            elif step_.outputs:
                for artifact, value in zip(step_.outputs, result, strict=True):
                    store[artifact] = value
            logger.info(
                "pipeline_step_completed",
                extra={"pipeline": self.name, "step": step_.name, "outputs": list(step_.outputs)},
            )
        return store


def build_default_pipeline() -> Pipeline:
    """Return the canonical ``fetch → validate → transform`` pipeline."""
    return Pipeline([fetch_step, validate_step, transform_step])


def run_pipeline(
    source: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience one-shot runner for the default pipeline.

    Args:
        source: Optional local CSV path or HTTP(S) URL. When ``None`` the
            deterministic synthetic generator is used.
        config: Optional full configuration mapping (``config/config.yaml``
            layout); loaded from disk when omitted.

    Returns:
        Catalog with ``raw_df``, ``fetch_stats``, ``validated_df``,
        ``validation_report``, ``graph`` and ``scaler``.
    """
    if config is None:
        from src.utils.config import load_config

        config = load_config()
    return build_default_pipeline().run(source=source, config=config)
