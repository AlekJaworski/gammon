"""Shared helpers + fixture loading for the gamrs Python test suite.

The suite is *tiered* (see ``pyproject.toml`` ``[tool.pytest.ini_options]``):

- ``smoke``  — fast essential correctness + regression guards. Touches every
  public API path once on tiny data; runs in well under a second. This is the
  tier you want in a pre-commit / fast-feedback loop.
- ``parity`` — numerical parity against the mgcv reference fixtures shared with
  the Rust ``tests/parity_*.rs`` battery. One fit per fixture per family, so it
  maximises statistical coverage (families × links × data shapes) for the cost
  of a few hundred small fits.
- ``slow``   — the large-n parity fixtures (n≥1000). Deselected by default to
  keep the headline run quick; run explicitly with ``-m slow``.

Run tiers:

    pytest tests/python                       # smoke + parity (default)
    pytest tests/python -m smoke              # fast tier only
    pytest tests/python -m "smoke or parity or slow"   # everything
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# tests/python/conftest.py -> tests/fixtures
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    """Load a parity fixture JSON by stem (no ``.json``)."""
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"fixture not present: {path.name}")
    with path.open() as fh:
        return json.load(fh)


def x_train_2d(fx: dict[str, Any]) -> np.ndarray:
    """``(n, d)`` float64 design from a fixture's ``inputs.x_train``."""
    return np.ascontiguousarray(np.asarray(fx["inputs"]["x_train"], dtype=np.float64))


def y_train(fx: dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(fx["inputs"]["y_train"], dtype=np.float64))


def max_rel_err(pred: np.ndarray, target: np.ndarray) -> float:
    """Max element-wise relative error against ``|target| + 1``.

    Identical denominator convention to the Rust ``max_rel_err`` in
    ``tests/parity_*.rs`` so the Python and Rust bars are directly comparable.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.max(np.abs(pred - target) / (np.abs(target) + 1.0)))


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    # Fixed seed: synthetic-data smoke tests must be deterministic.
    return np.random.default_rng(20260529)


@pytest.fixture(scope="session")
def toy_gaussian():
    """Small deterministic 1-D Gaussian dataset for smoke tests."""
    x = np.linspace(0.0, 1.0, 200)
    y = np.sin(2.0 * np.pi * x) + np.random.default_rng(0).normal(0.0, 0.1, x.size)
    return x, y
