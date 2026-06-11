"""Sinh-arcsinh (`shash`) — the NATIVE joint GAMLSS family.

Thin Python wrapper over the native ``fit_shash`` Rust driver, which fits the
four sinh-arcsinh linear predictors jointly by a dense block-Newton inner solve
under outer REML smoothing-parameter selection:

  - μ      (location)       via ``mu_terms``
  - log σ  (log-scale, τ)   via ``tau_terms`` (the ``logeb`` link, σ ≥ b)
  - ε      (skewness)       via ``eps_terms``
  - φ      (log-kurtosis)   via ``phi_terms``  (δ = exp(φ))

The native call returns a :class:`gamrs._gamrs_native.ShashGamFit`; this module
coerces inputs and resolves the typed term specs, then wraps the native fit so
``predict_eta`` / ``predict_params`` / ``predict_quantile`` accept the usual
DataFrame / ndarray / 1-D-vector inputs.

DISTINCT from :func:`gamrs.fit_quantile_lss` with ``shape="shash"`` (the helper
in :mod:`gamrs._shash`): that is a *two-stage* per-residual MLE used by the
quantile path — it fits μ(x) and σ(x) and then estimates a SINGLE global
(ε, δ) shape from the standardised residuals. ``fit_shash`` here is the genuine
joint GAMLSS: all four parameters are smooths of x, fit to the joint shash
likelihood, recovering mgcv ``gam(..., family=shash)`` to ~1e-2.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from . import _gamrs_native
from ._coerce import to_1d_array, to_2d_with_columns
from ._fitter import _resolve_term_cols
from ._low_level import _term_to_tuple


def _resolve_to_tuples(terms: Optional[Sequence[Any]], cols: list[str]) -> list[tuple]:
    """Resolve a (possibly None / empty) typed-term sequence to FFI tuples.

    ``None`` and ``[]`` both map to an empty list — an intercept-only predictor,
    which the native ``fit_shash`` handles directly.
    """
    if terms is None:
        return []
    resolved = [_resolve_term_cols(t, cols) for t in terms]
    return [_term_to_tuple(t) for t in resolved]


class ShashGamFit:
    """Fitted native sinh-arcsinh (`shash`) GAMLSS model.

    Wraps a native :class:`gamrs._gamrs_native.ShashGamFit`. Predicts the four
    linear predictors (:meth:`predict_eta`), the four fitted parameters
    (:meth:`predict_params`, ``(μ, σ, ε, δ)``), and any quantile
    (:meth:`predict_quantile`).

    Attributes:
      edf_: total effective degrees of freedom.
      laml_: the LAML / REML criterion at the optimum.
      converged_: True iff BOTH the outer REML ascent and the inner penalised
        Newton converged.
      rho_: selected log-smoothing-parameters (one per penalised block).
      block_p_: per-block coefficient counts ``(p_μ, p_τ, p_ε, p_φ)``.
      b_: the ``logeb`` bound used for the τ (log-scale) link.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.edf_ = float(inner.edf)
        self.laml_ = float(inner.laml)
        self.converged_ = bool(inner.converged) and bool(inner.inner_converged)
        self.rho_ = np.asarray(inner.rho, dtype=float)
        self.block_p_ = tuple(inner.block_p)
        self.b_ = float(inner.b)

    @staticmethod
    def _x(X: Any) -> np.ndarray:
        x2d, _ = to_2d_with_columns(X, None)
        return np.ascontiguousarray(x2d, dtype=np.float64)

    def predict_eta(self, X: Any) -> np.ndarray:
        """The four linear predictors on `X`: ``(n, 4)`` ``(η_μ, η_τ, η_ε, η_φ)``."""
        return np.asarray(self._inner.predict_eta(self._x(X)), dtype=float)

    def predict_params(self, X: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The four fitted parameters on `X` as ``(μ, σ, ε, δ)`` 1-D arrays.

        ``σ = exp(τ)`` via the ``logeb`` link; ``δ = exp(φ)``.
        """
        mu, sigma, eps, delta = self._inner.predict_params(self._x(X))
        return (
            np.asarray(mu, dtype=float).ravel(),
            np.asarray(sigma, dtype=float).ravel(),
            np.asarray(eps, dtype=float).ravel(),
            np.asarray(delta, dtype=float).ravel(),
        )

    def predict_quantile(self, X: Any, p: float) -> np.ndarray:
        """The fitted `p`-quantile per row of `X` (``(n,)``). ``0 < p < 1``."""
        return np.asarray(self._inner.predict_quantile(self._x(X), float(p)), dtype=float).ravel()

    def predict(self, X: Any, p: float = 0.5) -> np.ndarray:
        """Alias for :meth:`predict_quantile`; defaults to the median (p=0.5)."""
        return self.predict_quantile(X, p)


def fit_shash(
    X: Any,
    y: Any,
    mu_terms: Optional[Sequence[Any]] = None,
    tau_terms: Optional[Sequence[Any]] = None,
    eps_terms: Optional[Sequence[Any]] = None,
    phi_terms: Optional[Sequence[Any]] = None,
    b: float = 1e-2,
) -> ShashGamFit:
    """Fit a native sinh-arcsinh (`shash`) GAMLSS model.

    Models ``y ~ shash(μ(x), σ(x), ε(x), δ(x))`` with each parameter a smooth of
    the covariates, fit jointly to the shash likelihood by a dense block-Newton
    inner solve under outer REML. One fit yields every quantile via
    :meth:`ShashGamFit.predict_quantile`.

    This is the JOINT GAMLSS estimator — distinct from
    :func:`gamrs.fit_quantile_lss` with ``shape="shash"``, which is a two-stage
    per-residual MLE with a single global (ε, δ) shape.

    Args:
      X: ``(n, d)`` design (DataFrame / ndarray / 1-D vector).
      y: ``(n,)`` response.
      mu_terms: typed terms (`CrTerm` / `TeTerm` / …) for the location μ. An
        empty list / None is an intercept-only predictor.
      tau_terms: typed terms for the log-scale τ (``σ = exp(τ)``, ``logeb``
        link). Empty / None → intercept-only.
      eps_terms: typed terms for the skewness ε. Empty / None → intercept-only.
      phi_terms: typed terms for the log-kurtosis φ (``δ = exp(φ)``). Empty /
        None → intercept-only.
      b: the ``logeb`` bound for the τ link (``σ ≥ b``); mgcv default 1e-2.

    Returns:
      A :class:`ShashGamFit`.

    Note:
      v1 supports at most ONE smooth term per predictor (a single penalty per
      block); the native driver errors otherwise. Multi-smooth-per-predictor is
      a documented follow-up.
    """
    x2d, cols = to_2d_with_columns(X, None)
    y_arr = to_1d_array(y, name="y")
    if x2d.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {x2d.shape[0]} rows but y has {y_arr.shape[0]} elements"
        )
    x_c = np.ascontiguousarray(x2d, dtype=np.float64)
    y_c = np.ascontiguousarray(y_arr, dtype=np.float64)

    cols = list(cols)
    mu_tuples = _resolve_to_tuples(mu_terms, cols)
    tau_tuples = _resolve_to_tuples(tau_terms, cols)
    eps_tuples = _resolve_to_tuples(eps_terms, cols)
    phi_tuples = _resolve_to_tuples(phi_terms, cols)

    inner = _gamrs_native.fit_shash(
        mu_tuples, tau_tuples, eps_tuples, phi_tuples, x_c, y_c, float(b)
    )
    return ShashGamFit(inner)
