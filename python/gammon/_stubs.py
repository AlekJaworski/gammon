"""Forward-compat stubs that match v0.x's :mod:`mgcv_rust` exports
1-1 in shape but raise ``NotImplementedError`` for features gammon
doesn't yet wire.

Keeping these here (instead of inline in :mod:`gammon._fitter`) keeps
``_fitter.py`` under the 700-LOC project budget while still letting
consumer code do ``from gammon import TermContributions`` etc. without
``ImportError``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class TermContributions:
    """Result of ``Gam.predict(X, type='terms')``. Shape mirrors v0.x's
    namesake — gammon doesn't yet expose per-term coef indices through
    the bindings, so ``Gam.predict(type='terms')`` raises rather than
    constructing one of these directly. Fields are kept so consumer
    code can compile.
    """

    def __init__(self, intercept: float, contributions: Any, total: np.ndarray):
        self.intercept = intercept
        self.contributions = contributions
        self.total = total


class GamSummary:
    """Stub matching v0.x's ``GamSummary`` shape (intercept, scale, edf, λ).

    gammon surfaces the scalar values via :meth:`gammon.Gam.summary`; the
    per-term breakdown that v0.x's summary depends on is not yet wired
    here, but the dataclass-style attribute names match exactly so
    callers can read them off without code changes.
    """

    def __init__(
        self,
        family: str,
        link: str,
        n: int,
        intercept: float,
        scale: float,
        edf_total: float,
        lambda_: np.ndarray,
        converged: Optional[bool],
        n_iters: Optional[int],
    ):
        self.family = family
        self.link = link
        self.n = n
        self.intercept = intercept
        self.scale = scale
        self.edf_total = edf_total
        self.lambda_ = lambda_
        self.converged = converged
        self.n_iters = n_iters

    def __repr__(self) -> str:  # pragma: no cover — exercised manually
        return (
            f"GamSummary(family={self.family!r}, link={self.link!r}, n={self.n}, "
            f"intercept={self.intercept:.4g}, scale={self.scale:.4g}, "
            f"edf_total={self.edf_total:.4g}, lambda={self.lambda_}, "
            f"converged={self.converged}, n_iters={self.n_iters})"
        )


# GamPredictor moved to `gammon._predictor` — the fleshed-out, fit-time
# persistence-aware implementation. Importers that want the class should
# do `from gammon import GamPredictor` (or `from gammon._predictor import
# GamPredictor`). We intentionally avoid re-exporting here to dodge a
# circular import: `_predictor` itself imports from `_fitter`, and
# `_fitter` previously imported `GamPredictor` from this stub module.
