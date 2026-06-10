"""Gaussian location-scale (`gaulss`) — the first GAMLSS family.

Thin Python wrapper over the native ``fit_gaulss`` Rust driver, which fits the
two linear predictors (μ via ``mu_terms``, log σ via ``sigma_terms``) jointly
by orthogonal alternating Fisher scoring (block-diagonal Fisher information →
alternation of two single-predictor weighted-Gaussian REML fits). The native
call returns the two fitted blocks (each a native ``FittedGam``); ``GaulssFit``
composes them into location / scale / quantile predictions.

Distinct from :func:`gamrs.fit_quantile_lss` (a one-pass *two-stage* estimator):
``gaulss`` iterates to the joint location-scale MLE — the μ fit is reweighted
by 1/σ²(x) each pass (the GLS efficiency gain) and the scale uses the proper
Fisher-scoring likelihood. It matches mgcv ``gaulss`` to ~3-4 decimals.
"""

from __future__ import annotations

import statistics
from typing import Any, Optional, Sequence

import numpy as np

from . import _gamrs_native
from ._coerce import to_1d_array, to_2d_with_columns
from ._fitter import _resolve_term_cols
from ._low_level import CrTerm, _term_to_tuple
from ._quantile import _halve_term_k


class GaulssFit:
    """Fitted Gaussian location-scale model — ONE fit, ALL τ, no crossing.

    Wraps the μ and log σ block fits. Derives every quantile as
    ``q_τ(x) = μ̂(x) + σ̂(x)·Φ⁻¹(τ)`` (monotone in τ, σ̂ > 0 → no crossing).

    Attributes:
      n_iters_: outer alternation iterations to convergence.
      converged_: True iff the outer alternation hit the tolerance before the
        cap AND both block fits (μ and log σ) converged internally.
    """

    def __init__(self, loc: Any, scale: Any, n_iters: int, converged: bool):
        self._loc = loc
        self._scale = scale
        self.n_iters_ = int(n_iters)
        self.converged_ = bool(converged)

    def _x(self, X: Any) -> np.ndarray:
        x2d, _ = to_2d_with_columns(X, None)
        return np.ascontiguousarray(x2d, dtype=np.float64)

    def predict_loc(self, X: Any) -> np.ndarray:
        """Conditional mean μ̂(x)."""
        return np.asarray(self._loc.predict(self._x(X)), dtype=float).ravel()

    def predict_sigma(self, X: Any) -> np.ndarray:
        """Conditional standard deviation σ̂(x) = exp(η̂₂)."""
        return np.exp(np.asarray(self._scale.predict(self._x(X)), dtype=float).ravel())

    def predict_quantile(self, X: Any, tau: Any) -> np.ndarray:
        """`q_τ(x)` for one τ (``(n,)``) or many (``(n, n_τ)``); never crosses."""
        mu = self.predict_loc(X)
        sigma = self.predict_sigma(X)
        scalar = np.ndim(tau) == 0
        taus = np.atleast_1d(np.asarray(tau, dtype=float))
        if np.any((taus <= 0.0) | (taus >= 1.0)):
            raise ValueError("all tau must be in the open interval (0, 1)")
        z = np.array([statistics.NormalDist().inv_cdf(float(t)) for t in taus])
        q = mu[:, None] + sigma[:, None] * z[None, :]
        return q[:, 0] if scalar else q

    def predict(self, X: Any, tau: float = 0.5) -> np.ndarray:
        """Alias for :meth:`predict_quantile`; defaults to the median μ̂."""
        return self.predict_quantile(X, tau)


def fit_gaulss(
    X: Any,
    y: Any,
    mu_terms: Optional[Sequence[Any]] = None,
    sigma_terms: Optional[Sequence[Any]] = None,
    k: int = 10,
    k_scale: Optional[int] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> GaulssFit:
    """Fit a Gaussian location-scale (`gaulss`) GAMLSS model.

    Models `y ~ N(μ(x), σ(x)²)` with smooth μ(x) and σ(x), fit jointly. One
    fit yields every quantile via :meth:`GaulssFit.predict_quantile`.

    Args:
      X: ``(n, d)`` design (DataFrame / ndarray / 1-D vector).
      y: ``(n,)`` response.
      mu_terms: typed terms for the location (`CrTerm` / `TeTerm` / …); when
        None, one `CrTerm(col, k)` per input column.
      sigma_terms: typed terms for the log-scale; when None, mirrors the
        location columns with each basis dim ~halved (σ is usually flatter).
      k: location basis dim when `mu_terms` is None (default 10).
      k_scale: scale basis dim when `sigma_terms` is None and `mu_terms` is
        None. Defaults to ``max(3, k // 2)``.
      max_iter: outer alternation cap (default 50).
      tol: max-|Δlog σ| convergence tolerance (default 1e-6).

    Returns:
      A :class:`GaulssFit`.
    """
    x2d, cols = to_2d_with_columns(X, None)
    y_arr = to_1d_array(y, name="y")
    if x2d.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {x2d.shape[0]} rows but y has {y_arr.shape[0]} elements"
        )
    x_c = np.ascontiguousarray(x2d, dtype=np.float64)
    y_c = np.ascontiguousarray(y_arr, dtype=np.float64)

    n_cols = x2d.shape[1]
    ks = int(k_scale) if k_scale is not None else max(3, int(k) // 2)

    if mu_terms is None:
        mu_terms = [CrTerm(i, k=int(k)) for i in range(n_cols)]
        default_mu = True
    else:
        mu_terms = list(mu_terms)
        default_mu = False
    mu_resolved = [_resolve_term_cols(t, list(cols)) for t in mu_terms]

    if sigma_terms is not None:
        sig_resolved = [_resolve_term_cols(t, list(cols)) for t in sigma_terms]
    elif default_mu:
        sig_resolved = [CrTerm(i, k=ks) for i in range(n_cols)]
    else:
        sig_resolved = [_halve_term_k(t) for t in mu_resolved]

    mu_tuples = [_term_to_tuple(t) for t in mu_resolved]
    sig_tuples = [_term_to_tuple(t) for t in sig_resolved]

    loc, scale, n_iters, converged = _gamrs_native.fit_gaulss(
        x_c, y_c, mu_tuples, sig_tuples, int(max_iter), float(tol)
    )
    return GaulssFit(loc, scale, n_iters, converged)
