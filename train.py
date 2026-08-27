"""Notebook-compatible CLI shim delegating to :func:`src.training.train.main`.

Retained so the historical ``python train.py`` entrypoint keeps working;
all real logic lives in the refactored training package.
"""

from src.training.train import main

if __name__ == "__main__":
    main()
