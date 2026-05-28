"""Shared coercion helpers + the v0.x family-name → gamrs-native
mapping. Kept separate to keep ``_fitter.py`` under the 700-LOC
project budget.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


# Family name → (gamrs_native family_name, canonical_link). The native
# dispatcher accepts each value below; the link arg is informational
# at the Python wrapper layer.
FAMILY_TO_GAMRS: dict[str, tuple[str, str]] = {
    "gaussian": ("gaussian", "identity"),
    "binomial": ("bernoulli", "logit"),
    "bernoulli": ("bernoulli", "logit"),
    "poisson": ("poisson", "log"),
    "quasipoisson": ("quasipoisson", "log"),
    "quasibinomial": ("quasibinomial", "logit"),
    # mgcv's canonical link for Gamma is reciprocal (1/μ). The capital-G
    # "Gamma" matches R's mgcv default; lowercase "gamma" keeps the
    # historical log-link alias for backwards-compatibility with
    # pre-canonical-link callers.
    "Gamma": ("Gamma", "inverse"),
    "gamma": ("gamma", "log"),
    "inverse.gaussian": ("inverse_gaussian", "log"),
    "inverse_gaussian": ("inverse_gaussian", "log"),
    "negbin": ("negbin", "log"),
    "negative.binomial": ("negbin", "log"),
    "nb": ("negbin", "log"),
    "t-dist": ("tdist", "identity"),
    "scat": ("tdist", "identity"),
    "tweedie": ("tweedie", "log"),
    "tw": ("tweedie", "log"),
    "Tweedie": ("tweedie", "log"),
    "ocat": ("ocat", "identity"),
    "elf": ("elf", "identity"),
    "quantile": ("elf", "identity"),
}


def to_1d_array(y: Any, *, name: str) -> np.ndarray:
    """Coerce ``y`` to a 1-D ``float64`` contiguous ``ndarray``."""
    if hasattr(y, "to_numpy"):
        arr = y.to_numpy()
    elif hasattr(y, "to_pandas"):  # polars
        arr = y.to_pandas().to_numpy()
    else:
        arr = np.asarray(y)
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (or 2-D with one column); got shape={arr.shape}"
        )
    return arr


def to_2d_with_columns(
    X: Any, predictors: Optional[Sequence[str]]
) -> tuple[np.ndarray, list[str]]:
    """Coerce ``X`` to a 2-D ``float64`` ``ndarray`` + a column-name list."""
    if hasattr(X, "to_numpy") and hasattr(X, "columns"):
        cols = list(X.columns)
        if predictors is not None:
            arr = X[list(predictors)].to_numpy()
            cols = list(predictors)
        else:
            arr = X.to_numpy()
    elif hasattr(X, "to_pandas"):  # polars
        pdf = X.to_pandas()
        cols = list(pdf.columns)
        if predictors is not None:
            arr = pdf[list(predictors)].to_numpy()
            cols = list(predictors)
        else:
            arr = pdf.to_numpy()
    else:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = (
            list(predictors)
            if predictors is not None
            else [f"x{i}" for i in range(arr.shape[1])]
        )
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D; got ndim={arr.ndim} (shape={arr.shape})")
    if arr.shape[1] != len(cols):
        raise ValueError(
            f"X column count {arr.shape[1]} doesn't match column-name count {len(cols)}"
        )
    return arr, cols
