"""Centralised Modal secrets pattern for cloud training.

Never hard-code tokens or passwords: create each Modal ``Secret`` once from
your shell, then attach the handle to training functions via
``secrets=[...]`` so credentials arrive as environment variables at runtime.

Secrets managed here:

* ``aml-gnn-hf`` — ``HF_TOKEN`` for gated/private Hugging Face datasets.
* ``aml-gnn-neo4j`` — ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD``
  (+ optional ``NEO4J_DATABASE``) for graph-store reads/writes.

One-time setup (run locally)::

    modal secret create aml-gnn-hf HF_TOKEN=hf_xxx
    modal secret create aml-gnn-neo4j \\
        NEO4J_URI=bolt://host:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=xxx

Usage inside Modal functions::

    from infra.secrets_manager import hf_secret, neo4j_secret

    @app.function(image=image, secrets=[hf_secret(), neo4j_secret()])
    def train_remote(): ...

Locally the helpers raise a clear ``RuntimeError``; in CI without Modal they
resolve to ``None`` via ``optional_*`` so imports stay safe.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import modal
except ImportError:  # Modal is only required when deploying remotely.
    modal = None  # type: ignore[assignment]

#: Modal Secret holding the Hugging Face token (env: ``HF_TOKEN``).
HF_SECRET_NAME = "aml-gnn-hf"

#: Modal Secret holding Neo4j credentials (``NEO4J_URI/USER/PASSWORD``).
NEO4J_SECRET_NAME = "aml-gnn-neo4j"

#: Env var injected by ``HF_SECRET_NAME`` (read by ``huggingface_hub``).
HF_TOKEN_ENV = "HF_TOKEN"

#: Env vars injected by ``NEO4J_SECRET_NAME``.
NEO4J_URI_ENV = "NEO4J_URI"
NEO4J_USER_ENV = "NEO4J_USER"
NEO4J_PASSWORD_ENV = "NEO4J_PASSWORD"
NEO4J_DATABASE_ENV = "NEO4J_DATABASE"


def hf_secret() -> Any:
    """Return the Hugging Face ``modal.Secret`` handle.

    Raises:
        RuntimeError: If ``modal`` is not installed locally.
    """
    if modal is None:
        raise RuntimeError("Modal is required to resolve secrets; install 'modal'.")
    return modal.Secret.from_name(HF_SECRET_NAME)


def neo4j_secret() -> Any:
    """Return the Neo4j ``modal.Secret`` handle.

    Raises:
        RuntimeError: If ``modal`` is not installed locally.
    """
    if modal is None:
        raise RuntimeError("Modal is required to resolve secrets; install 'modal'.")
    return modal.Secret.from_name(NEO4J_SECRET_NAME)


def optional_hf_secret() -> Any | None:
    """Return the HF secret, or ``None`` when Modal is unavailable (CI-safe)."""
    if modal is None:
        return None
    return hf_secret()


def optional_neo4j_secret() -> Any | None:
    """Return the Neo4j secret, or ``None`` when Modal is unavailable (CI-safe)."""
    if modal is None:
        return None
    return neo4j_secret()


def training_secrets() -> list[Any]:
    """Return ``[hf, neo4j]`` secrets for training functions (CI-safe).

    Filters out ``None`` entries so callers can splat the result directly::

        @app.function(image=image, secrets=training_secrets())
    """
    return [s for s in (optional_hf_secret(), optional_neo4j_secret()) if s is not None]


def validate_local_env(require: bool = False) -> dict[str, bool]:
    """Check which credential env vars are present in the local shell.

    Args:
        require: Raise ``RuntimeError`` listing missing vars when ``True``.

    Returns:
        Mapping of env var name to presence flag (never leaks values).
    """
    keys = (HF_TOKEN_ENV, NEO4J_URI_ENV, NEO4J_USER_ENV, NEO4J_PASSWORD_ENV)
    status = {key: bool(os.environ.get(key)) for key in keys}
    if require:
        missing = [key for key, present in status.items() if not present]
        if missing:
            raise RuntimeError(
                f"Missing credential env vars: {missing}; "
                f"create them via 'modal secret create {HF_SECRET_NAME} ...' / "
                f"'modal secret create {NEO4J_SECRET_NAME} ...'"
            )
    return status


__all__ = [
    "HF_SECRET_NAME",
    "HF_TOKEN_ENV",
    "NEO4J_DATABASE_ENV",
    "NEO4J_PASSWORD_ENV",
    "NEO4J_SECRET_NAME",
    "NEO4J_URI_ENV",
    "NEO4J_USER_ENV",
    "hf_secret",
    "neo4j_secret",
    "optional_hf_secret",
    "optional_neo4j_secret",
    "training_secrets",
    "validate_local_env",
]
