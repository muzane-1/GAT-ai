"""Modal GPU image builder for cost-effective cloud training.

Decouples slow environment builds from training logic: build this image once
(``modal run infra.image_builder`` / ``modal build``) and reuse the cached
image from training functions in ``src/training/`` so GPU workers never pay
for ``pip`` installs or data transfers on every run.

Heavy dependencies pre-installed here:

* PyTorch (CUDA-enabled wheel, see ``CUDA_VERSION`` / ``TORCH_INDEX_URL``)
* PyTorch Geometric (``torch_geometric`` + compiled ``pyg_lib`` backend)
* Pandera (schema contract shared with ``src/pipeline/validation/``)
* Neo4j drivers (``neo4j`` package for graph-store reads/writes)

The image stays in ``infra/`` on purpose — ``src/`` (models/training logic)
is untouched and only *references* ``infra.image_builder.image``.
"""

from __future__ import annotations

from typing import Any

try:
    import modal
except ImportError:  # Modal is only required when building/deploying remotely.
    modal = None  # type: ignore[assignment]

#: Python runtime baked into the image (matches Dockerfile 3.11).
PYTHON_VERSION = "3.11"

#: CUDA release backing the Torch wheels. Modal GPU workers provide the
#: driver; the wheel must match (cu121 == CUDA 12.1, works on A10/T4/A100).
CUDA_VERSION = "12.1"

#: Pinned Torch matching ``torch>=2.2,<2.8`` in ``requirements.txt``.
TORCH_VERSION = "2.7.1"

#: CUDA wheel index for Torch (compatible with ``CUDA_VERSION`` above).
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"

#: Pre-compiled PyG backend wheels for Torch 2.7 + CUDA 12.1.
PYG_FIND_LINKS = "https://data.pyg.org/whl/torch-2.7.0+cu121.html"

#: Human-readable image tag used in logs / Modal dashboard.
IMAGE_NAME = "aml-gnn-gpu"

#: Fallback PyPI index kept when a custom ``index_url``/``find_links`` is used.
PYPI_EXTRA_INDEX_URL = "https://pypi.org/simple/"


def build_image() -> Any:
    """Build (or fetch from cache) the shared GPU training image.

    Returns:
        The cached ``modal.Image`` with CUDA Torch, PyG, Pandera and Neo4j.

    Raises:
        RuntimeError: If ``modal`` is not installed locally.
    """
    if modal is None:
        raise RuntimeError("Modal is required to build the image; install 'modal'.")

    torch_layer = modal.Image.debian_slim(python_version=PYTHON_VERSION).pip_install(
        f"torch=={TORCH_VERSION}",
        index_url=TORCH_INDEX_URL,
    )
    pyg_layer = torch_layer.pip_install(
        "torch_geometric>=2.5",
        "pyg_lib",
        find_links=PYG_FIND_LINKS,
        extra_index_url=PYPI_EXTRA_INDEX_URL,
    )
    full_layer = pyg_layer.pip_install(
        "pandera>=0.20",
        "neo4j>=5.0",
        "pandas>=2.0",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "PyYAML>=6.0",
        "pyarrow>=15.0",
    )
    return full_layer


def get_image() -> Any:
    """Return the shared image, building it lazily (cache-friendly)."""
    return build_image()


# Import-safe: ``None`` locally / in CI without Modal; built image remotely.
# Ternary form satisfies ruff SIM108 while keeping the fallback explicit.
image = get_image() if modal is not None else None


__all__ = [
    "CUDA_VERSION",
    "IMAGE_NAME",
    "PYG_FIND_LINKS",
    "PYTHON_VERSION",
    "TORCH_INDEX_URL",
    "TORCH_VERSION",
    "build_image",
    "get_image",
    "image",
]
