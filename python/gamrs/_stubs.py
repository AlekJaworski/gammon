"""Forward-compat stubs that match v0.x's :mod:`mgcv_rust` exports
1-1 in shape but raise ``NotImplementedError`` for features gamrs
doesn't yet wire.

Keeping these here (instead of inline in :mod:`gamrs._fitter`) keeps
``_fitter.py`` under the 700-LOC project budget while still letting
consumer code do ``from gamrs import TermContributions`` etc. without
``ImportError``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class TermContributions:
    """Result of ``Gam.predict(X, type='terms')``. Shape mirrors v0.x's
    namesake — gamrs doesn't yet expose per-term coef indices through
    the bindings, so ``Gam.predict(type='terms')`` raises rather than
    constructing one of these directly. Fields are kept so consumer
    code can compile.
    """

    def __init__(self, intercept: float, contributions: Any, total: np.ndarray):
        self.intercept = intercept
        self.contributions = contributions
        self.total = total


class GamSummary:
    """Compact summary of a fitted :class:`gamrs.Gam`.

    Returned by :meth:`Gam.summary`. Includes the per-smooth DataFrame
    (``smooths``) and top-level fit metadata. Pretty-prints in an
    mgcv-style block via ``repr()``. Shape matches v0.x's GamSummary
    1-1 so consumer code reading these fields off doesn't need to
    branch on which wrapper produced the summary.
    """

    def __init__(
        self,
        family: str,
        link: str,
        n_obs: int,
        intercept: float,
        intercept_response: float,
        smooths: Any,
        scale: float,
        deviance: float,
        r_squared: float,
        edf_total: float,
        converged: Optional[bool],
        n_iters: Optional[int],
    ):
        self.family = family
        self.link = link
        self.n_obs = n_obs
        self.intercept = intercept
        self.intercept_response = intercept_response
        self.smooths = smooths  # pd.DataFrame
        self.scale = scale
        self.deviance = deviance
        self.r_squared = r_squared
        self.edf_total = edf_total
        self.converged = converged
        self.n_iters = n_iters

    def __repr__(self) -> str:  # pragma: no cover — exercised manually
        lines = [
            f"Gam summary  family={self.family}  link={self.link}  n_obs={self.n_obs}",
            f"  intercept (link)     = {self.intercept:.6g}",
            f"  intercept (response) = {self.intercept_response:.6g}",
            f"  edf_total            = {self.edf_total:.4g}",
            f"  converged={self.converged}  n_iters={self.n_iters}",
        ]
        if not np.isnan(self.scale):
            lines.append(f"  scale (σ²)           = {self.scale:.6g}")
        if not np.isnan(self.deviance):
            lines.append(f"  deviance             = {self.deviance:.6g}")
        if not np.isnan(self.r_squared):
            lines.append(f"  R² (adj)             = {self.r_squared:.4f}")
        lines.append("  smooths:")
        try:
            for _, row in self.smooths.iterrows():
                lines.append(
                    f"    s({row['predictor']:>12s})  k={int(row['k']):>3d}  "
                    f"edf={row['edf']:>6.2f}  λ={row['lambda']:.3e}"
                )
        except Exception:  # pragma: no cover
            lines.append(f"    {self.smooths!r}")
        return "\n".join(lines)


# GamPredictor moved to `gamrs._predictor` — the fleshed-out, fit-time
# persistence-aware implementation. Importers that want the class should
# do `from gamrs import GamPredictor` (or `from gamrs._predictor import
# GamPredictor`). We intentionally avoid re-exporting here to dodge a
# circular import: `_predictor` itself imports from `_fitter`, and
# `_fitter` previously imported `GamPredictor` from this stub module.
