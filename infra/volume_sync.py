"""Cloud data volume + local-to-cloud sync entrypoint.

Decouples slow data transfers from GPU training: run the sync once from your
laptop/CI (``modal run infra.volume_sync``) so training workers mount a warm
``aml-gnn-data-vol`` Volume instead of re-uploading ``data/processed/`` on
every run.

Layout:

* Local source: ``data/processed/`` (see ``processed_dir`` in
  ``config/config.yaml``) — Parquet tables + ``transactions.pt`` PyG artifact.
* Cloud destination: ``modal.Volume`` named ``aml-gnn-data-vol``.
* Cloud mount point inside training functions: ``/data``
  (``VOLUME_MOUNT_PATH``).

Usage::

    modal run infra.volume_sync                    # sync data/processed -> /data
    modal run infra.volume_sync --dry-run          # list what would be synced
    modal run infra.volume_sync --local-dir data/raw --remote-path /raw
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:  # Modal is only required when syncing/deploying remotely.
    modal = None  # type: ignore[assignment]

#: Canonical cloud Volume holding processed AML artifacts.
VOLUME_NAME = "aml-gnn-data-vol"

#: Mount point of ``VOLUME_NAME`` inside Modal training functions.
VOLUME_MOUNT_PATH = "/data"

#: Local directory synced to the Volume (mirrors ``data.processed_dir``).
LOCAL_PROCESSED_DIR = Path("data/processed")

#: Remote prefix inside the Volume (``/`` == mount root ``/data``).
REMOTE_PATH = "/"


def get_volume() -> Any:
    """Return the shared data Volume, creating it on first use.

    Raises:
        RuntimeError: If ``modal`` is not installed locally.
    """
    if modal is None:
        raise RuntimeError("Modal is required to resolve a Volume; install 'modal'.")
    return modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def collect_local_files(local_dir: str | Path = LOCAL_PROCESSED_DIR) -> list[Path]:
    """List regular files under ``local_dir`` (sorted, for stable uploads).

    Raises:
        FileNotFoundError: If ``local_dir`` does not exist.
    """
    root = Path(local_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Local directory {root!s} not found; run the pipeline first so data/processed/ exists"
        )
    return sorted(p for p in root.rglob("*") if p.is_file())


if modal is not None:
    volume = get_volume()
    app = modal.App("aml-gnn-data-sync")

    @app.local_entrypoint()
    def sync(
        local_dir: str = str(LOCAL_PROCESSED_DIR),
        remote_path: str = REMOTE_PATH,
    ) -> None:
        """Sync ``local_dir`` into the cloud Volume (local entrypoint).

        Runs on your machine — not on GPU — via ``modal run infra.volume_sync``.
        Uses batched upload when available (single commit) and falls back to
        per-file ``add_local_file`` on older Modal clients.

        Args:
            local_dir: Local directory to upload (default ``data/processed``).
            remote_path: Destination prefix inside the Volume (default ``/``).
        """
        from src.utils.logger import get_logger

        logger = get_logger(__name__)
        root = Path(local_dir)
        files = collect_local_files(root)
        if not files:
            logger.warning("No files found under %s; nothing to sync", root)
            return

        vol = get_volume()
        prefix = remote_path.rstrip("/") or "/"
        logger.info(
            "Syncing %d file(s) from %s to volume %s:%s",
            len(files),
            root,
            VOLUME_NAME,
            prefix,
        )

        batch = getattr(vol, "batch_upload", None)
        if callable(batch):
            with batch() as uploader:
                for path in files:
                    target = f"{prefix}/{path.relative_to(root).as_posix()}".replace("//", "/")
                    uploader.put_file(str(path), target)
                    logger.info("Queued %s -> %s", path, target)
        else:  # Older Modal client: one committed put per file.
            for path in files:
                target = f"{prefix}/{path.relative_to(root).as_posix()}".replace("//", "/")
                vol.add_local_file(path, target)  # type: ignore[attr-defined]
                logger.info("Uploaded %s -> %s", path, target)
        vol.commit() if hasattr(vol, "commit") else None
        logger.info("Volume %s sync complete (%d file(s))", VOLUME_NAME, len(files))

else:  # Import-safe locally / in CI without Modal credentials.
    volume = None
    app = None

    def sync(
        local_dir: str = str(LOCAL_PROCESSED_DIR),
        remote_path: str = REMOTE_PATH,
    ) -> None:
        """Report a useful error when Modal is not installed locally."""
        raise RuntimeError("Modal is required to sync the Volume; install 'modal'.")


__all__ = [
    "LOCAL_PROCESSED_DIR",
    "REMOTE_PATH",
    "VOLUME_MOUNT_PATH",
    "VOLUME_NAME",
    "app",
    "collect_local_files",
    "get_volume",
    "sync",
    "volume",
]
