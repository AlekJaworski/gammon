"""Minimal coercion facade over :func:`gammon._gammon_native.fit`.

The native binding is strict (it wants 1-D float64 contiguous arrays
for x and y). This wrapper handles the common 2-D ``(n, 1)`` /
non-float64 / non-contiguous inputs by coercing them once before
delegating, and exposes the returned :class:`FittedGam` directly.

Mirrors the role of :mod:`mgcv_rust._low_level` for v0.x users.

Multi-smooth additive fits (epic 94b/94c) are surfaced via the typed
:class:`CrTerm` / :class:`ReTerm` / :class:`TeTerm` term classes plus
the :func:`fit_additive` helper. The FFI boundary still uses tuples
internally, but the Python surface stays typed (no string-keyed knobs
leak to user code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from . import _gammon_native


def _ensure_1d_float64(arr: Any, *, name: str) -> np.ndarray:
    """Coerce ``arr`` to a 1-D ``float64`` contiguous ``ndarray``.

    2-D inputs of shape ``(n, 1)`` are reshaped to ``(n,)`` — the
    common notebook case where a DataFrame column was selected.
    """
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.reshape(-1)
    if a.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (or a 2-D column vector); got ndim={a.ndim} (shape={a.shape})"
        )
    return a


class GAM:
    """Coerce-and-forward wrapper around :func:`gammon._gammon_native.fit`.

    Use this when you want a single-predictor 1-D fit without the
    DataFrame plumbing of :class:`gammon.Gam`. Method signatures match
    the native ``fit`` function plus light input coercion.
    """

    __slots__ = ("_fitted", "family_name", "design", "k")

    def __init__(
        self,
        family: str = "gaussian",
        *,
        k: int = 10,
        design: str = "cr",
    ) -> None:
        self.family_name = family
        self.k = int(k)
        self.design = design
        self._fitted: Optional[Any] = None

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        weights: Any = None,
        k: Optional[int] = None,
        design: Optional[str] = None,
        **family_kwargs: Any,
    ) -> "GAM":
        x_arr = _ensure_1d_float64(x, name="x")
        y_arr = _ensure_1d_float64(y, name="y")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"x and y must have the same length; got x={x_arr.shape[0]}, y={y_arr.shape[0]}"
            )
        w_arr = (
            _ensure_1d_float64(weights, name="weights") if weights is not None else None
        )
        self._fitted = _gammon_native.fit(
            self.family_name,
            x_arr,
            y_arr,
            weights=w_arr,
            k=int(k if k is not None else self.k),
            design=design if design is not None else self.design,
            **family_kwargs,
        )
        return self

    def _require_fitted(self) -> Any:
        if self._fitted is None:
            raise RuntimeError("GAM has not been fitted yet — call .fit() first.")
        return self._fitted

    def predict(self, x: Any, *, scale: str = "link") -> np.ndarray:
        f = self._require_fitted()
        x_arr = _ensure_1d_float64(x, name="x")
        if scale == "link":
            return np.asarray(f.predict(x_arr))
        if scale == "response":
            return np.asarray(f.predict_response(x_arr))
        raise ValueError(f"scale must be 'link' or 'response', got {scale!r}")

    def predict_ci(
        self, x: Any, *, level: float = 0.95, scale: str = "response"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self._require_fitted()
        x_arr = _ensure_1d_float64(x, name="x")
        mean, lo, hi = f.predict_ci(x_arr, float(level), scale)
        return np.asarray(mean), np.asarray(lo), np.asarray(hi)

    def predict_diff(
        self, x_a: Any, x_b: Any, *, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self._require_fitted()
        a = _ensure_1d_float64(x_a, name="x_a")
        b = _ensure_1d_float64(x_b, name="x_b")
        diff, lo, hi = f.predict_diff(a, b, float(level))
        return np.asarray(diff), np.asarray(lo), np.asarray(hi)

    def vcov(self) -> np.ndarray:
        return np.asarray(self._require_fitted().vcov())

    # --- Forward common getters as attributes for sklearn-ish ergonomics. -----

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self._require_fitted().beta)

    @property
    def scale_(self) -> float:
        return float(self._require_fitted().scale)

    @property
    def edf_total_(self) -> float:
        return float(self._require_fitted().edf_total)

    @property
    def rho_(self) -> Union[float, np.ndarray]:
        """Fitted log smoothing parameters. Returns a float for single-
        smooth fits (v0.x-compatible scalar) and an ndarray for multi-
        smooth `Additive` fits."""
        rho = np.asarray(self._require_fitted().rho)
        if rho.shape == (1,):
            return float(rho[0])
        return rho

    @property
    def lambda_(self) -> np.ndarray:
        # `lambda` is a Python keyword — fetch via getattr to access the
        # native PyO3 getter of the same name.
        return np.asarray(getattr(self._require_fitted(), "lambda"))

    @property
    def n_iters_(self) -> int:
        return int(self._require_fitted().n_iters)

    @property
    def converged_(self) -> bool:
        return bool(self._require_fitted().converged)


# =============================================================================
# Typed term dataclasses for multi-smooth `fit_additive` (94b/94c).
# =============================================================================


@dataclass(frozen=True)
class CrTerm:
    """Univariate CR-spline term `s(x_col, bs='cr', k=k)`.

    `col` is the integer column index into the design `x`. `k` is the
    spline basis dim (defaults to 10, matching mgcv).
    """
    col: int
    k: int = 10


@dataclass(frozen=True)
class CrStableTerm:
    """CR + sum-to-zero + StableReparam rotation term."""
    col: int
    k: int = 10


@dataclass(frozen=True)
class ReTerm:
    """Random-effect term `s(x_col, bs='re')` — one-hot encoded grouping."""
    col: int


@dataclass(frozen=True)
class TeTerm:
    """Anisotropic 2-margin tensor product term `te(x_col_a, x_col_b)` —
    two smoothing parameters per term (one per margin). CR margins.

    `cols` is the pair `(col_a, col_b)` of column indices into the
    design `x`. `k` is the pair `(k_a, k_b)` of marginal basis dims
    (defaults to ``(10, 10)``).
    """
    cols: Tuple[int, int]
    k: Tuple[int, int] = (10, 10)


# Sum type at the Python boundary. Type-checked closed set — adding a
# new term kind extends this union (a library-controlled change).
Term = Union[CrTerm, CrStableTerm, ReTerm, TeTerm]


def _term_to_tuple(term: Term) -> tuple:
    """Convert a typed `Term` to the tuple form `_gammon_native.fit_additive`
    expects at the FFI boundary. Strings live ONLY here, between the
    Python typed surface and the Rust enum — they never leak in either
    direction."""
    if isinstance(term, CrTerm):
        return (int(term.col), "cr", int(term.k))
    if isinstance(term, CrStableTerm):
        return (int(term.col), "cr_stable", int(term.k))
    if isinstance(term, ReTerm):
        return (int(term.col), "re")
    if isinstance(term, TeTerm):
        return (
            (int(term.cols[0]), int(term.cols[1])),
            "te",
            (int(term.k[0]), int(term.k[1])),
        )
    raise TypeError(
        f"unknown term type {type(term).__name__}; expected one of "
        "CrTerm / CrStableTerm / ReTerm / TeTerm"
    )


def fit_additive(
    family: str,
    x: Any,
    y: Any,
    terms: Sequence[Term],
    *,
    weights: Any = None,
) -> Any:
    """Fit a multi-smooth additive GAM `y ~ s(x_{c_0}) + s(x_{c_1}) + ...`.

    `x` must be a 2-D array of shape ``(n_obs, n_input_dims)``. `y` is
    1-D of length `n_obs`. `terms` is a sequence of typed term objects
    — :class:`CrTerm`, :class:`CrStableTerm`, :class:`ReTerm`, or
    :class:`TeTerm`. Tensor terms (:class:`TeTerm`) provide two
    smoothing parameters per term and read two columns of `x`.

    Returns the native :class:`FittedGam`. Use
    :attr:`FittedGam.rho` / :attr:`FittedGam.lambda` / :attr:`FittedGam.edf_per_term`
    (all length = number of smoothing params, i.e. 1 per univariate term,
    2 per tensor term) for per-term diagnostics.

    `family` accepts the same strings as :func:`_gammon_native.fit` minus
    the shape-managed families (tdist, scat, negbin, tweedie, ocat, elf,
    quantile) which are restricted to single-smooth in 94b.
    """
    x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x_arr.ndim != 2:
        raise ValueError(
            f"x must be 2-D (n_obs, n_input_dims); got ndim={x_arr.ndim} (shape={x_arr.shape})"
        )
    y_arr = _ensure_1d_float64(y, name="y")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"x and y must have the same number of rows; got x.shape[0]={x_arr.shape[0]}, "
            f"y.shape[0]={y_arr.shape[0]}"
        )
    w_arr = (
        _ensure_1d_float64(weights, name="weights") if weights is not None else None
    )
    term_tuples = [_term_to_tuple(t) for t in terms]
    if not term_tuples:
        raise ValueError("terms must be non-empty")
    return _gammon_native.fit_additive(family, x_arr, y_arr, term_tuples, weights=w_arr)
