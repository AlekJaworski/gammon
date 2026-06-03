"""Sklearn-style ``Gam`` wrapper — drop-in for ``mgcv_rust.Gam``.

Mirrors the v0.x ``mgcv_rust.Gam`` API surface 1-1 for the basic
single-smooth use case. See :mod:`gamrs` for the full migration story
and the list of features that raise ``NotImplementedError`` (with a
pointer back at v0.x for the ones not yet wired).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

from . import _gamrs_native
from ._coerce import FAMILY_TO_GAMRS, to_1d_array, to_2d_with_columns
from ._low_level import (
    CrTerm,
    CrStableTerm,
    ParametricTerm,
    ReTerm,
    TeMultiTerm,
    TeTerm,
    Term,
    TiTerm,
    TpsTerm,
    _term_to_tuple,
)
from ._stubs import GamSummary, TermContributions  # noqa: F401

ArrayLike = Any  # avoid hard dep on pandas/polars typing


def _normal_quantile(p: float) -> float:
    """Φ⁻¹(p) without a scipy dependency. Beasley-Springer-Moro
    rational approximation (Moro 1995) — accurate to ~7 sig figs in
    (0, 1), good enough for Wald-CI z-scores."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = (-2.0 * np.log(p)) ** 0.5
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = (-2.0 * np.log(1.0 - p)) ** 0.5
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


def _resolve_col(col: Any, col_names: Sequence[str], term_name: str) -> int:
    """Resolve one column reference (int or string) to an int index.

    Strings are looked up in ``col_names`` (the resolved DataFrame /
    ``predictors=`` list). Raises ``ValueError`` with the offending term
    name when a string doesn't match any column.
    """
    if isinstance(col, str):
        try:
            return col_names.index(col)
        except ValueError:
            raise ValueError(
                f"{term_name}: column {col!r} not found in design; "
                f"available columns: {list(col_names)}"
            ) from None
    return int(col)


def _resolve_term_cols(term: Term, col_names: Sequence[str]) -> Term:
    """Rebuild a term with string column references replaced by their int
    indices in ``col_names``. Int-only terms pass through unchanged."""
    if isinstance(term, CrTerm):
        return CrTerm(col=_resolve_col(term.col, col_names, "CrTerm"), k=term.k)
    if isinstance(term, CrStableTerm):
        return CrStableTerm(
            col=_resolve_col(term.col, col_names, "CrStableTerm"), k=term.k
        )
    if isinstance(term, ReTerm):
        return ReTerm(col=_resolve_col(term.col, col_names, "ReTerm"))
    if isinstance(term, ParametricTerm):
        return ParametricTerm(
            col=_resolve_col(term.col, col_names, "ParametricTerm")
        )
    if isinstance(term, TeTerm):
        return TeTerm(
            cols=(
                _resolve_col(term.cols[0], col_names, "TeTerm"),
                _resolve_col(term.cols[1], col_names, "TeTerm"),
            ),
            k=term.k,
        )
    if isinstance(term, TeMultiTerm):
        return TeMultiTerm(
            cols=tuple(_resolve_col(c, col_names, "TeMultiTerm") for c in term.cols),
            k=term.k,
        )
    if isinstance(term, TiTerm):
        return TiTerm(
            cols=tuple(_resolve_col(c, col_names, "TiTerm") for c in term.cols),
            k=term.k,
        )
    if isinstance(term, TpsTerm):
        return TpsTerm(
            cols=tuple(_resolve_col(c, col_names, "TpsTerm") for c in term.cols),
            k=term.k,
        )
    raise TypeError(
        f"unknown term type {type(term).__name__}; expected one of "
        "CrTerm / CrStableTerm / ReTerm / TeTerm / TeMultiTerm / TiTerm / TpsTerm"
    )


# =============================================================================
# Gam — sklearn-style wrapper. Mirrors mgcv_rust.Gam.
# =============================================================================


class Gam:
    """Sklearn-style GAM with smoothing-parameter selection by REML or fREML.

    Fits one of ten families (see ``family=``). Supports single 1-D smooths,
    additive multi-smooth (``y ~ s(x0) + s(x1) + …``), n-margin tensor
    products (``te()`` / ``ti()``), thin-plate splines, and random effects
    via the typed-term API.

    Two common ways to specify the smooth structure:

    .. code-block:: python

        # Implicit: one CR smooth per column of X
        g = Gam(family="gaussian").fit(X, y)

        # Explicit: typed terms with string or int column references
        from gamrs import CrTerm, TeTerm
        g = Gam(terms=[CrTerm("x0", k=10), CrTerm("x1", k=15)]).fit(X, y)
        g = Gam(terms=[TeTerm(cols=("x0", "x1"), k=(8, 8))]).fit(X, y)

    For GLM families at large n, pass ``method="fREML"`` for the
    mgcv R ``bam()`` optimiser (Fellner-Schall multiplicative λ updates
    with single-step IRLS). See ``docs/perf.md`` for the trade-off.

    Compatible with the v0.x ``mgcv_rust.Gam`` API surface; a textual
    substitution of ``from mgcv_rust import Gam`` → ``from gamrs import Gam``
    is the intended migration path."""

    # ---- Class constants matching v0.x ------------------------------- #
    INTERCEPT = "__constant__"

    # -------------------------- Constructor --------------------------- #

    def __init__(
        self,
        predictors: Optional[Sequence[str]] = None,
        target: Optional[str] = None,
        min_k: int = 3,
        k_default: int = 10,
        k_index_margin: float = 0.05,
        max_k_auto: int = 50,
        auto_k_max_iter: int = 5,
        knots_increase_ratio: float = 1.5,
        min_points_to_save: int = 100,
        max_points_to_save: int = 1000,
        method: Optional[str] = None,
        family: str = "gaussian",
        link: Optional[str] = None,
        df: Optional[float] = None,
        tweedie_p: Optional[float] = None,
        negbin_theta: Optional[float] = None,
        r: Optional[int] = None,
        term_k_mapping: Optional[dict[str, int]] = None,
        term_pc_mapping: Optional[dict[str, float]] = None,
        predictor_basis_map: Optional[dict[str, str]] = None,
        consider_categorical: bool = False,
        auto_k: bool = False,
        discrete: bool = False,
        # gamrs-only knobs:
        design: str = "cr",
        terms: Optional[Sequence[Term]] = None,
        **kwargs: Any,
    ) -> None:
        if family not in FAMILY_TO_GAMRS:
            raise ValueError(
                f"family={family!r} is not supported by gamrs; "
                f"supported: {sorted(FAMILY_TO_GAMRS.keys())}"
            )
        self.predictors: Optional[list[str]] = (
            list(predictors) if predictors is not None else None
        )
        self.target: str = target or "y"
        self.min_k = int(min_k)
        self.k_default = int(k_default)
        self.k_index_margin = float(k_index_margin)
        self.max_k_auto = int(max_k_auto)
        self.auto_k_max_iter = int(auto_k_max_iter)
        self.knots_increase_ratio = knots_increase_ratio
        self.min_points_to_save = min_points_to_save
        self.max_points_to_save = max_points_to_save
        self.df = df
        self.tweedie_p = tweedie_p
        self.negbin_theta = negbin_theta
        self.r = int(r) if r is not None else None
        # Quantile (ELF) config — set by gamrs._quantile.fit_quantile so the
        # single Gam.fit lands at the right (τ, σ). Default None → native τ=0.5.
        self._elf_tau: Optional[float] = None
        self._elf_sigma: Optional[float] = None
        if family == "ocat" and self.r is None:
            raise ValueError(
                "family='ocat' requires r=K (number of ordered categories, K >= 3)"
            )
        if family == "ocat" and self.r < 3:
            raise ValueError(f"ocat requires r >= 3, got r={self.r}")
        if r is not None and family != "ocat":
            raise ValueError(
                f"r= is only meaningful for family='ocat'; got family={family!r}"
            )

        if method is None:
            method = "fREML" if family in ("scat", "t-dist") else "REML"
        self.method = method
        self.family = family
        gamrs_family, canonical_link = FAMILY_TO_GAMRS[family]
        self._gamrs_family = gamrs_family
        self.link = link if link is not None else canonical_link
        if self.link != canonical_link:
            warnings.warn(
                f"gamrs currently uses the canonical link {canonical_link!r} for "
                f"family={family!r}; the requested link={self.link!r} is ignored.",
                UserWarning,
                stacklevel=2,
            )
            self.link = canonical_link

        # df validation for tdist
        if df is not None and family in ("t-dist", "scat"):
            if df < 2.0:
                raise ValueError(f"t-dist df must be >= 2.0, got {df}")
            if df > 100.0:
                raise ValueError(
                    f"t-dist df must be <= 100.0, got {df}. Use df ∈ [2, 100]."
                )
        elif df is not None and family not in ("t-dist", "scat"):
            raise ValueError(
                f"df= is only meaningful for family='t-dist' (or its mgcv alias "
                f"'scat'); got family={family!r}"
            )

        self.term_k_mapping: dict[str, int] = dict(term_k_mapping or {})
        self.term_pc_mapping: dict[str, float] = dict(term_pc_mapping or {})
        self.predictor_basis_map: dict[str, str] = dict(predictor_basis_map or {})
        self.consider_categorical = consider_categorical
        self.auto_k = auto_k
        self.discrete = bool(discrete)
        self.design = design
        # Typed-term API alternative to predictors=/predictor_basis_map=.
        # When both are passed, terms= wins and the others are ignored.
        self.terms: Optional[list[Term]] = list(terms) if terms is not None else None

        # API-compat knobs accepted from mgcv_rust source-compatible code
        # but currently no-ops in gamrs. Warn once per Gam() construction so
        # users aren't quietly mismatched against their mgcv expectations.
        if self.discrete:
            warnings.warn(
                "discrete=True is accepted for mgcv_rust source compatibility "
                "but is currently a no-op in gamrs — the dense PIRLS path is "
                "already faster than mgcv_rust 0.23's discrete-binning path at "
                "n ≤ 1M (see docs/perf.md). The fit will proceed without "
                "discrete binning.",
                UserWarning,
                stacklevel=2,
            )
        nthreads_arg = kwargs.pop("nthreads", None)
        if nthreads_arg is not None:
            warnings.warn(
                f"nthreads={nthreads_arg!r} is accepted for mgcv_rust source "
                "compatibility but gamrs doesn't expose a thread count "
                "directly — set OPENBLAS_NUM_THREADS / MKL_NUM_THREADS in the "
                "environment instead. The fit will proceed using whatever the "
                "BLAS thread pool is currently configured for.",
                UserWarning,
                stacklevel=2,
            )

        # Forward-compat: stash any remaining unknown kwargs without
        # erroring so a textual mgcv_rust → gamrs substitution doesn't
        # blow up at construction time.
        if kwargs:
            self._unknown_kwargs = kwargs

        # Filled at fit time:
        self._fitted = self.X = self.y = self.sample_weight = None
        self._effective_predictors = self._original_predictors = None
        self.dropped_predictors_ = {}
        self._k_used = None
        # Subset-view state. None on a fitted model means "use all terms".
        # Set by __getitem__; consulted by predict() to mask un-selected
        # term blocks of the lpmatrix.
        self._subset_mask: Optional[set] = None
        # Auto-k diagnostics. _auto_k_trace records (iteration, predictor,
        # k, edf, k_index, grew) per inner refit; exposed via auto_k_trace_.
        self._auto_k_iterations: int = 0
        self._auto_k_trace: list[dict[str, Any]] = []

    # ------------------------ helpers --------------------------------- #

    def _require_fitted(self) -> Any:
        if self._fitted is None:
            raise RuntimeError("Model has not been fitted yet — call .fit() first.")
        return self._fitted

    def _family_kwargs_for_native(self) -> dict[str, Any]:
        """Build the gamrs-native ``fit(...)`` family-specific kwargs from
        the v0.x constructor knobs (df, tweedie_p, negbin_theta, r, ...)."""
        out: dict[str, Any] = {}
        if self.family in ("negbin", "negative.binomial", "nb") and self.negbin_theta is not None:
            out["theta"] = float(self.negbin_theta)
        if self.family in ("t-dist", "scat") and self.df is not None:
            # gamrs's tdist takes nu (ν); df is its v0.x alias.
            out["nu"] = float(self.df)
        if self.family in ("tweedie", "tw", "Tweedie") and self.tweedie_p is not None:
            out["tweedie_p"] = float(self.tweedie_p)
        if self.family == "ocat":
            out["r"] = int(self.r)
        # Quantile (ELF): emit τ (always — the native default is 0.5) and σ
        # when configured, so the single Gam.fit lands at the right (τ, σ)
        # directly. `gamrs._quantile.fit_quantile` sets these before fit()
        # so it fits ONCE (no fit-then-replace), matching mgcv_rust.
        if self.family in ("quantile", "elf"):
            out["tau"] = float(getattr(self, "_elf_tau", None) or 0.5)
            elf_sigma = getattr(self, "_elf_sigma", None)
            if elf_sigma is not None:
                out["elf_sigma"] = float(elf_sigma)
        return out

    # ------------------------ fit / predict --------------------------- #

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Any = None) -> "Gam":
        """Fit the GAM. Drop-in for ``mgcv_rust.Gam.fit``.

        ``X`` may be a DataFrame / 2-D ndarray / 1-D ndarray. Multi-column
        X dispatches to the additive multi-smooth path (one CR term per
        column by default; ``predictor_basis_map`` lets you switch a
        column to ``"re"``). For the typed-term API, pass ``terms=`` to
        the constructor — that path bypasses predictor-name resolution.

        The ``method`` constructor argument selects the outer optimiser:
        ``"REML"`` / ``None`` (default) → damped Newton on the REML
        score; ``"fREML"`` → Wood & Fasiolo (2017) Fellner-Schall
        multiplicative updates (mgcv R ``bam()`` equivalent; cheaper
        per-iter, wins on GLM families at large n).
        """
        # Set/restore the outer-algorithm override for this fit only.
        # Only touches the thread-local when method != "REML" (gamrs's
        # default) — `"REML"` is what an unspecified method becomes via
        # the constructor default and shouldn't clobber any external
        # override (e.g. set by the bench / sweep scripts).
        wants_override = self.method is not None and self.method != "REML"
        if wants_override:
            try:
                _gamrs_native.set_outer_algorithm(self.method)
            except ValueError as e:
                raise ValueError(
                    f"method={self.method!r} not recognised; "
                    f"supported: 'REML' (default), 'fREML'"
                ) from e
        try:
            return self._fit_impl(X, y, sample_weight)
        finally:
            if wants_override:
                _gamrs_native.set_outer_algorithm(None)

    def _fit_impl(self, X: ArrayLike, y: ArrayLike, sample_weight: Any) -> "Gam":
        X_arr, cols = to_2d_with_columns(X, self.predictors)
        y_arr = to_1d_array(y, name="y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]} elements"
            )

        if self.predictors is None:
            self.predictors = cols
        self._original_predictors = list(cols)
        self.dropped_predictors_ = {}

        # Auto-drop columns with n_unique == 1 — they contribute zero signal
        # and would otherwise blow up the CR-basis k≥3 check. Matches
        # mgcv_rust 0.23.0's silent-drop behaviour (which in turn matches
        # mgcv R's QR rank detection at fit time).
        #
        # Only applies on the predictors= / predictor_basis_map= path; when
        # the user passes `terms=` they're being fully explicit about which
        # columns map to which smooths, and silently dropping one of their
        # terms would be more confusing than the original "k≥3" error.
        if self.terms is None:
            keep_mask = np.array(
                [int(np.unique(X_arr[:, i]).size) > 1 for i in range(X_arr.shape[1])],
                dtype=bool,
            )
            if not keep_mask.all():
                for i, keep in enumerate(keep_mask):
                    if not keep:
                        name = cols[i]
                        const_val = float(X_arr[0, i])
                        self.dropped_predictors_[name] = const_val
                        warnings.warn(
                            f"predictor {name!r} is constant (n_unique=1, "
                            f"value={const_val}); dropping from the design — "
                            "adds no signal and would otherwise fail the "
                            "spline k≥3 check. Available on "
                            "`dropped_predictors_`.",
                            UserWarning,
                            stacklevel=3,
                        )
                X_arr = X_arr[:, keep_mask]
                cols = [c for c, k in zip(cols, keep_mask) if k]
                if X_arr.shape[1] == 0:
                    raise ValueError(
                        "all predictor columns are constant — nothing to fit. "
                        f"Dropped: {list(self.dropped_predictors_)!r}"
                    )

        self._effective_predictors = list(cols)
        self.X = X_arr
        self.y = y_arr

        # Coerce sample_weight.
        if sample_weight is not None:
            w_arr = to_1d_array(sample_weight, name="sample_weight")
            if w_arr.shape[0] != X_arr.shape[0]:
                raise ValueError(
                    f"sample_weight has {w_arr.shape[0]} elements but X has "
                    f"{X_arr.shape[0]} rows"
                )
            self.sample_weight = w_arr
        else:
            self.sample_weight = None

        x_2d = np.ascontiguousarray(X_arr, dtype=np.float64)

        # Build the term list — either from self.terms (typed API) or
        # derived from columns + predictor_basis_map (v0.x API).
        if self.terms is not None:
            # Resolve any string col references against the now-known
            # column names. Ints pass through unchanged.
            term_objs = [_resolve_term_cols(t, list(cols)) for t in self.terms]
        else:
            term_objs = self._build_terms_from_columns(X_arr, cols)

        if self.auto_k:
            self._auto_fit_k(x_2d, y_arr, term_objs)
        else:
            self._auto_k_iterations = 0
            self._auto_k_trace = []
            self._single_fit(x_2d, y_arr, term_objs)
        return self

    # ---- single-fit + auto-k helpers --------------------------------- #

    def _single_fit(
        self, x_2d: np.ndarray, y_arr: np.ndarray, term_objs: list[Term]
    ) -> None:
        """Run one native fit at the given term list. Updates `_fitted`
        and `_k_used`; doesn't mutate `_auto_k_*`."""
        family_kw = self._family_kwargs_for_native()
        self._k_used = [
            t.k if isinstance(t, (CrTerm, CrStableTerm)) else
            (t.k[0] * t.k[1] if isinstance(t, TeTerm) else 0)
            for t in term_objs
        ]
        # Single-smooth → fast path (preserves byte-equivalence with
        # pre-multi-smooth fits + the existing parity tests).
        if len(term_objs) == 1 and isinstance(term_objs[0], (CrTerm, ReTerm, CrStableTerm)):
            t = term_objs[0]
            if isinstance(t, CrTerm):
                design, k_arg = "cr", t.k
            elif isinstance(t, CrStableTerm):
                design, k_arg = "cr_stable", t.k
            else:  # ReTerm
                design, k_arg = "re", 2  # k unused for re; pass a placeholder
            self._fitted = _gamrs_native.fit(
                self._gamrs_family,
                x_2d,
                y_arr,
                weights=self.sample_weight,
                k=int(k_arg),
                design=design,
                **family_kw,
            )
            return
        # Multi-smooth → additive dispatch.
        term_tuples = [_term_to_tuple(t) for t in term_objs]
        self._fitted = _gamrs_native.fit_additive(
            self._gamrs_family,
            x_2d,
            y_arr,
            term_tuples,
            weights=self.sample_weight,
            **family_kw,
        )

    @staticmethod
    def _k_index(x_col: np.ndarray, resid: np.ndarray) -> float:
        """mgcv's `k.check` statistic: sort residuals by `x_col` and form

            k_index = Σ (r_(i+1) − r_(i))² / (2 · Var(r) · (n − 1))

        Under the null (residuals i.i.d. given the smooth) `E[diff²] =
        2·Var(r)` and the statistic converges to 1. Values below 1 mean
        consecutive residuals (after sorting by x) are more similar than
        chance — i.e. leftover structure → basis is too small. See Wood
        (2017) §5.9. Returns 1.0 when Var(r) ≈ 0 (perfect fit).
        """
        n = int(resid.size)
        if n < 2:
            return 1.0
        var_r = float(np.var(resid))
        if var_r < 1e-12:
            return 1.0
        order = np.argsort(x_col, kind="stable")
        diffs = np.diff(resid[order])
        return float(np.sum(diffs ** 2) / (2.0 * var_r * (n - 1)))

    def _auto_fit_k(
        self,
        x_2d: np.ndarray,
        y_arr: np.ndarray,
        term_objs: list[Term],
    ) -> None:
        """Iteratively refit, growing `k` for any CR smooth whose
        residuals still show structure along its predictor's axis.

        Per non-frozen CR term j, each iteration:
        1. Compute residual k-index (see `_k_index`).
        2. If `k_index < 1 − k_index_margin`, grow
           `k_j ← ceil(k_j · knots_increase_ratio)`, capped at
           `min(n_unique(x_j) − 1, max_k_auto)`.

        Stops when no term grew, every term hit its cap, or
        `auto_k_max_iter` iterations have run.

        ReTerm / TeTerm and any term whose user-name is in
        `term_k_mapping` are treated as frozen — k-index still recorded
        in the trace for diagnostics, never grown.
        """
        preds = self._effective_predictors or []
        # Per-CR-term cap = min(n_unique − 1, max_k_auto), floored at min_k.
        caps: list[int] = []
        for t in term_objs:
            if isinstance(t, (CrTerm, CrStableTerm)):
                col = t.col if isinstance(t.col, int) else preds.index(t.col)
                n_unique = int(np.unique(x_2d[:, col]).size)
                caps.append(min(max(n_unique - 1, self.min_k), self.max_k_auto))
            else:
                caps.append(0)  # frozen (re / te) — never grown

        threshold = 1.0 - self.k_index_margin
        self._auto_k_trace = []
        current: list[Term] = list(term_objs)
        for iteration in range(self.auto_k_max_iter):
            self._single_fit(x_2d, y_arr, current)
            fitted = np.asarray(self._fitted.predict(x_2d), dtype=float)
            resid = y_arr - fitted

            grew = False
            all_capped = True
            for j, t in enumerate(current):
                user_name = preds[j] if j < len(preds) else f"term_{j}"
                frozen = (
                    not isinstance(t, (CrTerm, CrStableTerm))
                    or user_name in self.term_k_mapping
                )
                if not isinstance(t, (CrTerm, CrStableTerm)):
                    continue  # k_index not meaningful for re / te bases
                col = t.col if isinstance(t.col, int) else preds.index(t.col)
                k_idx = self._k_index(x_2d[:, col], resid)
                k_before = t.k
                term_grew = False
                if not frozen and k_idx < threshold:
                    new_k = math.ceil(k_before * self.knots_increase_ratio)
                    capped_k = min(new_k, caps[j])
                    if capped_k > k_before:
                        current[j] = type(t)(col=t.col, k=int(capped_k))
                        term_grew = True
                        grew = True
                if not frozen and t.k < caps[j]:
                    all_capped = False
                self._auto_k_trace.append({
                    "iteration": iteration,
                    "predictor": user_name,
                    "k": k_before,
                    "k_index": k_idx,
                    "grew": term_grew,
                })

            self._auto_k_iterations = iteration + 1
            if not grew or all_capped:
                break

    def _build_terms_from_columns(
        self, X_arr: np.ndarray, cols: Sequence[str]
    ) -> list[Term]:
        """Translate the v0.x predictors / predictor_basis_map / term_k_mapping
        signature into a list of typed terms. One term per column.
        """
        out: list[Term] = []
        for col_idx, pname in enumerate(cols):
            bs_override = self.predictor_basis_map.get(pname)
            if bs_override in ("parametric", "linear"):
                out.append(ParametricTerm(col=col_idx))
                continue
            if bs_override == "re":
                out.append(ReTerm(col=col_idx))
                continue
            if bs_override is not None and bs_override not in ("cr",):
                warnings.warn(
                    f"predictor_basis_map[{pname!r}]={bs_override!r} not yet "
                    "supported by gamrs; falling back to 'cr'.",
                    UserWarning,
                    stacklevel=3,
                )
            k = int(self.term_k_mapping.get(pname, self.k_default))
            n_unique = int(np.unique(X_arr[:, col_idx]).size)
            k = max(2, min(k, max(2, n_unique - 1)))
            out.append(CrTerm(col=col_idx, k=k))
        return out

    def _coerce_predict_X(self, X: ArrayLike) -> np.ndarray:
        X_arr, _ = to_2d_with_columns(X, self._effective_predictors)
        expected = (
            len(self._effective_predictors) if self._effective_predictors is not None else None
        )
        if expected is not None and X_arr.shape[1] != expected:
            raise ValueError(
                f"predict X has {X_arr.shape[1]} columns; expected {expected} "
                f"(matching fit-time predictors {self._effective_predictors!r})"
            )
        return np.ascontiguousarray(X_arr, dtype=np.float64)

    def predict(
        self,
        X: ArrayLike,
        scale: str = "response",
        type: Optional[str] = None,
    ) -> Union[np.ndarray, TermContributions]:
        """Predict on the requested scale. Drop-in for v0.x.

        ``scale``: ``'response'`` (inv-linked), ``'link'`` (η), or
        ``'deviation'`` (subset views only — η contribution of just the
        selected terms, intercept dropped).

        ``type='terms'`` returns per-term contributions as an
        ``(n_rows, n_terms)`` ndarray (intercept column dropped). Each
        column j is the η contribution of term j on the link scale.
        """
        f = self._require_fitted()
        if type not in (None, "terms"):
            raise ValueError(f"type must be None or 'terms', got {type!r}")
        if scale not in ("response", "link", "deviation"):
            raise ValueError(
                f"scale must be 'response', 'link', or 'deviation', got {scale!r}"
            )
        x = self._coerce_predict_X(X)
        beta = np.asarray(f.beta)
        ranges = f.term_col_ranges()

        if type == "terms":
            lp = np.asarray(f.evaluate_lpmatrix(x))
            return np.column_stack(
                [lp[:, start:end] @ beta[start:end] for (start, end) in ranges]
            )

        # Subset view: mask un-selected term blocks before β·lp.
        if self._subset_mask is not None:
            lp = np.asarray(f.evaluate_lpmatrix(x))
            masked = self._apply_subset_mask(lp, ranges, scale)
            eta = masked @ beta
        else:
            if scale == "deviation":
                raise ValueError(
                    "scale='deviation' is only meaningful on subset views — "
                    "use gam[[<predictors>]].predict(X, scale='deviation')."
                )
            eta = np.asarray(f.predict(x))

        if scale in ("link", "deviation"):
            return eta
        # response scale — inverse-link η elementwise (use native path
        # when possible to match its inverse-link conventions exactly).
        if self._subset_mask is None:
            return np.asarray(f.predict_response(x))
        # Subset path: apply the same inverse link manually.
        link = (self.link or "identity").lower()
        if link == "log":
            return np.exp(eta)
        if link == "logit":
            return 1.0 / (1.0 + np.exp(-eta))
        return eta  # identity

    def _apply_subset_mask(
        self,
        lp: np.ndarray,
        ranges: Sequence[tuple[int, int]],
        scale: str,
    ) -> np.ndarray:
        """Zero out columns belonging to terms NOT in ``self._subset_mask``.

        Intercept (column 0) is kept iff ``"__constant__" in self._subset_mask``,
        OR if ``scale != "deviation"`` and no intercept directive was given.
        On ``scale='deviation'`` the intercept is dropped, leaving the pure
        marginal effect of the selected terms.
        """
        mask = self._subset_mask  # type: ignore[union-attr]
        preds = self._effective_predictors or []
        out = lp.copy()
        intercept_selected = self.INTERCEPT in mask
        if scale == "deviation" or not intercept_selected:
            out[:, 0] = 0.0
        for term_idx, (start, end) in enumerate(ranges):
            user_name = preds[term_idx] if term_idx < len(preds) else f"term_{term_idx}"
            if user_name not in mask:
                out[:, start:end] = 0.0
        return out

    def predict_ci(
        self,
        X: ArrayLike,
        alpha: Optional[float] = None,
        n_samples: int = 1000,
        predictor: Optional[str] = None,
        seed: int = 42,
        *,
        level: float = 0.95,
        scale: str = "response",
    ) -> tuple[np.ndarray, ...]:
        """Pointwise CI for predictions. Drop-in for v0.x.

        Default returns ``(mean, lo, hi)``; legacy ``alpha=`` returns
        ``(lo, hi)`` with a ``DeprecationWarning``. Computed via gamrs's
        cached vcov + Wald formula (closed-form, not posterior sampling
        — ``n_samples`` / ``seed`` / ``predictor`` are accepted for
        signature compat and ignored).
        """
        f = self._require_fitted()
        if scale not in ("response", "link", "deviation"):
            raise ValueError(
                f"scale must be 'response', 'link', or 'deviation', got {scale!r}"
            )
        if scale == "deviation" and self._subset_mask is None:
            raise ValueError(
                "scale='deviation' is only meaningful on subset views — "
                "create one with gam[[\"predictor\"]] first, then call "
                "predict_ci(..., scale='deviation') on the view."
            )

        deprecated = alpha is not None
        if deprecated:
            warnings.warn(
                "predict_ci(alpha=...) is deprecated; use predict_ci(level=1-alpha) "
                "and unpack the new (mean, lo, hi) return.",
                DeprecationWarning,
                stacklevel=2,
            )
            effective_level = 1.0 - float(alpha)
        else:
            if not 0.0 < level < 1.0:
                raise ValueError(f"level must be in (0, 1), got {level}")
            effective_level = float(level)

        # Subset views are η-component summaries; response-scale CIs don't
        # have a coherent definition on them. Reject early with guidance.
        if self._subset_mask is not None and scale == "response":
            raise ValueError(
                "predict_ci(scale='response') is only defined for full-model "
                "views. On a subset view (gam[[...]]), the prediction is an "
                "η-component of a single smooth's contribution — use "
                "scale='link' (with intercept) or scale='deviation' (without)."
            )

        x = self._coerce_predict_X(X)

        # Subset view: native predict_ci can't apply the term mask, so
        # compute Wald CI from the masked lpmatrix here (same math as
        # partial_effect's CI branch).
        if self._subset_mask is not None:
            lp = np.asarray(f.evaluate_lpmatrix(x))
            ranges = f.term_col_ranges()
            masked = self._apply_subset_mask(lp, ranges, scale=scale)
            vcov = np.asarray(f.vcov())
            var_eta = np.einsum("ij,jk,ik->i", masked, vcov, masked)
            beta = np.asarray(f.beta)
            mean = masked @ beta
            z = _normal_quantile(0.5 + 0.5 * effective_level)
            sd = np.sqrt(np.maximum(var_eta, 0.0))
            lo, hi = mean - z * sd, mean + z * sd
            if deprecated:
                return np.asarray(lo), np.asarray(hi)
            return np.asarray(mean), np.asarray(lo), np.asarray(hi)

        # Full-model path — let the native CI do the work.
        scale_arg = "link" if scale == "link" else "response"
        mean, lo, hi = f.predict_ci(x, effective_level, scale_arg)
        if deprecated:
            return np.asarray(lo), np.asarray(hi)
        return np.asarray(mean), np.asarray(lo), np.asarray(hi)

    def predict_diff(
        self,
        from_X: ArrayLike,
        to_X: ArrayLike,
        level: Optional[float] = None,
        broadcast: str = "none",
        n_samples: int = 1000,
        seed: int = 42,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Δ = predict(to_X) − predict(from_X) on η scale. Drop-in for v0.x.

        ``level=None`` → bare ndarray; float → (diff, lo, hi) triple.
        ``broadcast`` ∈ {none, from, to}. Non-identity-link raises
        ``NotImplementedError`` (matches v0.x)."""
        f = self._require_fitted()
        if (self.link or "identity").lower() not in ("identity", ""):
            raise NotImplementedError(
                f"predict_diff is only implemented for identity-link models "
                f"(this model uses link={self.link!r}). For non-identity links "
                "sample posteriors at to_X and from_X separately and take the "
                "response-scale difference per draw."
            )
        if broadcast not in ("none", "from", "to"):
            raise ValueError(
                f"broadcast must be 'none', 'from', or 'to', got {broadcast!r}"
            )

        from_arr = self._coerce_predict_X(from_X)
        to_arr = self._coerce_predict_X(to_X)
        if broadcast == "none":
            if from_arr.shape[0] != to_arr.shape[0]:
                raise ValueError(
                    f"broadcast='none' requires equal row counts, got "
                    f"from_X={from_arr.shape[0]} rows and to_X={to_arr.shape[0]} rows"
                )
        elif broadcast == "from":
            if from_arr.shape[0] != 1:
                raise ValueError(
                    f"broadcast='from' requires from_X to have exactly 1 row, "
                    f"got {from_arr.shape[0]}"
                )
        else:  # "to"
            if to_arr.shape[0] != 1:
                raise ValueError(
                    f"broadcast='to' requires to_X to have exactly 1 row, "
                    f"got {to_arr.shape[0]}"
                )

        if level is None:
            # Bare diff — call gamrs's predict and subtract.
            eta_to = np.asarray(f.predict(to_arr))
            eta_from = np.asarray(f.predict(from_arr))
            if eta_to.shape != eta_from.shape:
                # Broadcast manually.
                if eta_from.shape == (1,):
                    return eta_to - eta_from[0]
                return eta_to[0] - eta_from
            return eta_to - eta_from

        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        diff, lo, hi = f.predict_diff(to_arr, from_arr, float(level))
        return np.asarray(diff), np.asarray(lo), np.asarray(hi)

    # --------------------- sklearn-style fitted attrs ------------------ #

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self._require_fitted().beta)

    @property
    def intercept_(self) -> float:
        return float(self._require_fitted().beta[0])

    @property
    def intercept_response_(self) -> float:
        # Apply the link's inverse to the intercept. Identity = identity.
        eta = float(self._require_fitted().beta[0])
        link = (self.link or "identity").lower()
        if link == "log":
            return float(np.exp(eta))
        if link == "logit":
            return float(1.0 / (1.0 + np.exp(-eta)))
        return eta  # identity (or unknown link → η as fallback)

    @property
    def vcov_(self) -> np.ndarray:
        return np.asarray(self._require_fitted().vcov())

    @property
    def scale_(self) -> float:
        return float(self._require_fitted().scale)

    @property
    def edf_total_(self) -> float:
        return float(self._require_fitted().edf_total)

    @property
    def edf_(self) -> np.ndarray:
        """Per-smooth EDF. Multi-smooth fits return one entry per term;
        single-smooth returns length-1 (matches v0.x's per-smooth shape)."""
        return np.asarray(self._require_fitted().edf_per_term)

    @property
    def lambda_(self) -> np.ndarray:
        # `lambda` is a Python keyword — fetch via getattr to access the
        # native PyO3 getter of the same name.
        return np.asarray(getattr(self._require_fitted(), "lambda"))

    @property
    def rho_(self) -> Union[float, np.ndarray]:
        """Fitted log smoothing parameters. Length-1 for single-smooth
        fits (returned as a float for v0.x compatibility); multi-smooth
        fits return the full ndarray."""
        rho = np.asarray(self._require_fitted().rho)
        if rho.shape == (1,):
            return float(rho[0])
        return rho

    @property
    def converged_(self) -> Optional[bool]:
        return bool(self._require_fitted().converged)

    @property
    def n_iters_(self) -> int:
        return int(self._require_fitted().n_iters)

    @property
    def n_outer_iter_(self) -> Optional[int]:
        """v0.x alias for ``n_iters_``."""
        return self.n_iters_

    @property
    def reml_value_(self) -> float:
        return float(self._require_fitted().reml_value)

    @property
    def fit_stats_(self) -> dict:
        """Diagnostic counters captured during the fit.

        Keys: ``outer_iterations``, ``line_search_trials``,
        ``no_refresh_attempts``, ``no_refresh_hits``,
        ``inner_pirls_calls``, ``inner_pirls_iterations_total``,
        ``pirls_iters_per_call``, ``no_refresh_hit_rate``.
        """
        return dict(self._require_fitted().fit_stats)

    @property
    def feature_names_in_(self) -> np.ndarray:
        return np.array(self._effective_predictors or [])

    @property
    def n_features_in_(self) -> int:
        return len(self._effective_predictors or [])

    @property
    def k_(self) -> np.ndarray:
        """Per-smooth k vector. Length = number of terms."""
        if self._k_used is None:
            return np.array([], dtype=int)
        if isinstance(self._k_used, int):
            return np.array([self._k_used], dtype=int)
        return np.array(self._k_used, dtype=int)

    @property
    def bs_(self) -> np.ndarray:
        """Per-smooth basis kind. Length = number of terms."""
        if self.terms is not None:
            return np.array(
                [
                    "te" if isinstance(t, TeTerm) else
                    "re" if isinstance(t, ReTerm) else
                    "cr_stable" if isinstance(t, CrStableTerm) else "cr"
                    for t in self.terms
                ],
                dtype=object,
            )
        if self._effective_predictors is None:
            return np.array([self.design], dtype=object)
        return np.array(
            [self.predictor_basis_map.get(p, self.design) for p in self._effective_predictors],
            dtype=object,
        )

    @property
    def auto_k_trace_(self) -> Any:
        """Per-(iteration, predictor) trace of the auto-k loop.

        Returns a pandas DataFrame with columns
        ``['iteration', 'predictor', 'k', 'k_index', 'grew']`` (one row
        per term per iteration). Empty when ``auto_k=False`` or no
        CR smooths were grown. Use to diagnose runaway / no-growth
        regimes when tuning ``k_index_margin`` /
        ``knots_increase_ratio`` / ``max_k_auto``.
        """
        if not self._auto_k_trace:
            return []
        try:
            import pandas as pd
        except ImportError:
            return list(self._auto_k_trace)
        return pd.DataFrame(self._auto_k_trace)

    @property
    def ocat_theta_(self) -> np.ndarray:
        """Converged log-gap thresholds for ocat. Length `r - 2` — one
        threshold per gap between adjacent categories above the first.
        The full `r + 1` category boundary vector is
        `α = [-∞, -1, -1 + exp(θ₀), -1 + exp(θ₀) + exp(θ₁), …, +∞]`.
        """
        if self.family != "ocat":
            raise AttributeError("ocat_theta_ is only available for family='ocat'")
        return np.asarray(self._require_fitted().shape_params)

    @property
    def shape_params_(self) -> np.ndarray:
        """Family-specific fitted shape parameters. Empty for fixed-shape
        families (Gaussian, Bernoulli, Poisson, etc.). See family docs
        for the per-family layout (ocat → log-gap θ, t-dist → [log ν, log σ²]).
        """
        return np.asarray(self._require_fitted().shape_params)

    # ----------------------- Methods that match v0.x ----------------- #

    def get_lambdas(self) -> np.ndarray:
        return self.lambda_

    def get_coefficients(self) -> np.ndarray:
        return self.coef_

    def get_vcov(self) -> np.ndarray:
        return self.vcov_

    def get_design_matrix(self) -> np.ndarray:
        """Lpmatrix for the training X. ``(n_train, p)`` ndarray with
        column 0 = intercept and columns 1..p = per-term blocks.
        See :meth:`get_term_indices` for column ranges per term."""
        f = self._require_fitted()
        if self.X is None:
            raise RuntimeError(
                "Training X wasn't retained on this instance (subset view "
                "or deserialized model). Call evaluate_lpmatrix(X) with "
                "your X instead."
            )
        return np.asarray(f.evaluate_lpmatrix(self.X))

    def evaluate_lpmatrix(self, X: ArrayLike) -> np.ndarray:
        """Build the design matrix at new X. ``(n_new, p)`` ndarray with
        column 0 = intercept and columns 1..p = per-term blocks. Useful
        for custom posterior sampling, partial predictions, or plugging
        into downstream uncertainty pipelines."""
        f = self._require_fitted()
        x = self._coerce_predict_X(X)
        return np.asarray(f.evaluate_lpmatrix(x))

    def get_term_indices(self) -> list[tuple[str, int, int]]:
        """Per-term column ranges into the lpmatrix.

        Returns a list of ``(predictor_name, first, last_inclusive)``
        tuples in predictor order. ``first`` / ``last`` are 0-based
        indices into the ``(n, p)`` lpmatrix returned by
        :meth:`evaluate_lpmatrix`. The intercept (column 0) is NOT
        included — it sits at index 0 of every lpmatrix.
        """
        f = self._require_fitted()
        ranges = f.term_col_ranges()
        names = self._effective_predictors or [
            f"term_{i}" for i in range(len(ranges))
        ]
        return [
            (names[i] if i < len(names) else f"term_{i}", start, end - 1)
            for i, (start, end) in enumerate(ranges)
        ]

    def get_edf_df(self) -> Any:
        """Per-term EDF as a pandas DataFrame with columns
        ``['predictor', 'edf']``. Sum (plus intercept dof=1) ≈ edf_total."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "get_edf_df() needs pandas; install with `pip install pandas`."
            ) from exc
        f = self._require_fitted()
        edf = np.asarray(f.edf_per_term)
        names = self._effective_predictors or [f"term_{i}" for i in range(len(edf))]
        return pd.DataFrame({"predictor": names, "edf": edf})

    def get_posterior_samples(
        self, X: ArrayLike, n_samples: int = 1000, seed: int = 42
    ) -> np.ndarray:
        """Draw `n_samples` posterior η predictions at X.

        β draws ~ N(β̂, vcov) → η_s = lp · β_s for each draw. Returns shape
        ``(n_samples, n_rows)``. Use for posterior intervals on derived
        quantities, calibrated uncertainty propagation, etc.
        """
        f = self._require_fitted()
        x = self._coerce_predict_X(X)
        lp = np.asarray(f.evaluate_lpmatrix(x))
        beta = np.asarray(f.beta)
        vcov = np.asarray(f.vcov())
        rng = np.random.default_rng(seed)
        beta_samples = rng.multivariate_normal(beta, vcov, size=int(n_samples))
        return beta_samples @ lp.T  # (n_samples, n_rows)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return ``(n, 2)`` probability matrix for binomial; ``(n, R)``
        for ocat (one column per ordered category).
        """
        if self.family in ("binomial", "bernoulli"):
            p = self.predict(X, scale="response")
            return np.column_stack([1.0 - p, p])
        if self.family == "ocat":
            # ocat: predict returns η, the thresholds in native_state give
            # P(Y <= k). We expose them via the native FittedGam's beta /
            # ocat_theta — but the cleanest path is to ask the native
            # for inv-linked μ which already encodes the per-category
            # probabilities. gamrs's ocat predict_response returns the
            # (n, R) matrix directly — see native python.rs.
            f = self._require_fitted()
            x = self._coerce_predict_X(X)
            # The ocat native predict_response returns flat (n*R,); reshape.
            r = int(self.r)
            mu = np.asarray(f.predict_response(x))
            if mu.ndim == 1 and mu.size % r == 0:
                mu = mu.reshape(-1, r)
            return mu
        raise NotImplementedError(
            f"predict_proba() is not wired for family={self.family!r}."
        )

    def partial_effect(
        self,
        predictor: str,
        grid_n: int = 100,
        level: Optional[float] = 0.95,
    ) -> Any:
        """Marginal effect of one smooth on a grid.

        Returns a pandas DataFrame with columns ``['x', 'mean']`` (and
        ``['lo', 'hi']`` if ``level`` is given). The grid spans the
        training range of the requested predictor; other predictors are
        held at their training medians. The mean is on the η scale
        (the subset view drops the intercept, matching ``scale='deviation'``).
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "partial_effect() needs pandas; install with `pip install pandas`."
            ) from exc
        self._require_fitted()
        if predictor not in (self._effective_predictors or []):
            raise KeyError(
                f"unknown predictor {predictor!r}; known: {self._effective_predictors!r}"
            )
        if self.X is None:
            raise RuntimeError(
                "partial_effect needs the training X to span the grid + median "
                "other predictors; not retained on this Gam (e.g. after a "
                "deserialize). Refit and retry."
            )

        col_idx = self._effective_predictors.index(predictor)  # type: ignore[union-attr]
        x_col = self.X[:, col_idx]
        x_grid = np.linspace(float(np.min(x_col)), float(np.max(x_col)), int(grid_n))
        # Build a (grid_n, n_features) X with the grid in `col_idx` and the
        # training median in every other column.
        medians = np.median(self.X, axis=0)
        X_grid = np.tile(medians, (int(grid_n), 1))
        X_grid[:, col_idx] = x_grid

        view = self[[predictor]]
        mean = view.predict(X_grid, scale="deviation")

        if level is None:
            return pd.DataFrame({"x": x_grid, "mean": mean})

        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")

        # CI via Wald on the masked lpmatrix: var(η_subset) =
        # (lp_masked · vcov · lp_masked.T).diag()
        f = self._fitted
        lp = np.asarray(f.evaluate_lpmatrix(X_grid))
        ranges = f.term_col_ranges()
        masked = view._apply_subset_mask(lp, ranges, scale="deviation")
        vcov = np.asarray(f.vcov())
        # diagonal of lp_masked · vcov · lp_masked.T, vectorised
        var_eta = np.einsum("ij,jk,ik->i", masked, vcov, masked)
        z = _normal_quantile(0.5 + 0.5 * level)
        sd = np.sqrt(np.maximum(var_eta, 0.0))
        return pd.DataFrame(
            {"x": x_grid, "mean": mean, "lo": mean - z * sd, "hi": mean + z * sd}
        )

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "plot() is not yet wired in gamrs. Use mgcv_rust.Gam.plot() or "
            "build a matplotlib figure from partial_effect data."
        )

    def summary(self) -> GamSummary:
        """Compact mgcv-style fit summary. Returns a :class:`GamSummary`
        with the per-smooth table and top-level fit metadata. ``repr()``
        formats it in a mgcv-style block. Gaussian fits also populate
        ``scale``, ``deviance``, and adjusted ``r_squared``.
        """
        f = self._require_fitted()
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "summary() requires pandas. pip install pandas"
            ) from exc

        names = self._effective_predictors or [f"term_{i}" for i in range(len(self.edf_))]
        smooths_df = pd.DataFrame(
            {
                "predictor": list(names),
                "k": list(self.k_),
                "edf": list(self.edf_),
                "lambda": list(self.lambda_),
            }
        )

        # Gaussian-only fit metrics, computed from fitted μ at training X.
        scale = float("nan")
        deviance = float("nan")
        r_sq = float("nan")
        if self.family == "gaussian" and self.X is not None and self.y is not None:
            x_train = np.ascontiguousarray(self.X, dtype=np.float64)
            fitted = np.asarray(f.predict_response(x_train), dtype=float)
            resid = self.y - fitted
            deviance = float(np.sum(resid ** 2))
            total_edf = float(self.edf_.sum()) + 1.0  # +1 for intercept
            dof = max(len(self.y) - total_edf, 1.0)
            scale = deviance / dof
            ss_tot = float(np.sum((self.y - self.y.mean()) ** 2))
            if ss_tot > 0:
                r_sq = 1.0 - (deviance / ss_tot) * ((len(self.y) - 1.0) / dof)

        return GamSummary(
            family=self.family,
            link=self.link,
            n_obs=int(f.n),
            intercept=float(self.intercept_),
            intercept_response=float(self.intercept_response_),
            smooths=smooths_df,
            scale=scale,
            deviance=deviance,
            r_squared=r_sq,
            edf_total=float(f.edf_total),
            converged=bool(f.converged),
            n_iters=int(f.n_iters),
        )

    # ----------------------- Persistence (save/load) ----------------- #
    # Bulk of the on-disk format lives in `_persistence.py`.

    def serialize(self) -> bytes:
        """Compact binary bytes (MAGIC | VERSION | LEN | bincode body).

        Round-trips bit-for-bit through :meth:`deserialize`: predictions
        after a reload are FP-identical to the original fit. ~3-5× smaller
        than the JSON form. Use this for production deployment; use
        :meth:`to_json` for human-debuggable / `jq`-able output. Pair with
        :meth:`save` / :meth:`load` to write to disk with the wrapper-side
        metadata included.
        """
        return bytes(self._require_fitted().serialize())

    @classmethod
    def deserialize(cls, payload: bytes) -> "Gam":
        """Rebuild from native bytes only — wrapper-side metadata is
        defaulted. Use :meth:`Gam.load` for the metadata-aware path."""
        return cls._wrap_native_fitted(_gamrs_native.FittedGam.deserialize(payload))

    def to_json(self) -> str:
        """Serialize to a plain UTF-8 JSON string (unframed,
        human-debuggable). Carries the same fitted state as
        :meth:`serialize` — β, vcov, knots, centring, reparam — so
        predictions after a JSON round-trip are numerically identical
        (mod f64 round-tripping through decimal representation, which
        ``serde_json`` performs losslessly).

        Pair with :meth:`from_json`. Useful for diffing two fits, hand
        inspection, or piping through ``jq``. For production deployment
        prefer :meth:`serialize` (3-5× smaller, faster to decode, and
        version-framed).
        """
        return self._require_fitted().serialize_json()

    @classmethod
    def from_json(cls, payload: str) -> "Gam":
        """Rebuild from a JSON string produced by :meth:`to_json`."""
        return cls._wrap_native_fitted(_gamrs_native.FittedGam.deserialize_json(payload))

    @classmethod
    def _wrap_native_fitted(cls, native_fitted: Any) -> "Gam":
        """Shared post-deserialize plumbing for both byte and JSON paths.

        Bypasses ``__init__`` (a deserialized model has no constructor
        args available) and defaults the wrapper-side metadata that the
        native fit doesn't carry. ``Gam.save`` / ``Gam.load`` are the
        metadata-aware path; this one drops the predictor names, term
        list, etc."""
        gam = cls.__new__(cls)
        gam._fitted = native_fitted
        gam.__dict__.update(
            predictors=None, _effective_predictors=None,
            _original_predictors=None, dropped_predictors_={},
            X=None, y=None, sample_weight=None, _k_used=None,
            family="gaussian", link="identity", _gamrs_family="gaussian",
            method="REML", target="y", design="cr",
            term_k_mapping={}, term_pc_mapping={}, predictor_basis_map={},
            consider_categorical=False, auto_k=False, discrete=False,
            df=None, tweedie_p=None, negbin_theta=None, r=None,
            terms=None, _subset_mask=None,
        )
        return gam

    def save(self, path: Union[str, "os.PathLike[str]"]) -> None:
        """Save to disk; see :mod:`gamrs._persistence` for the format."""
        from ._persistence import save_gam
        save_gam(self, path)

    @classmethod
    def load(cls, path: Union[str, "os.PathLike[str]"]) -> "Gam":
        """Load a Gam from a file written by :meth:`save`."""
        from ._persistence import load_gam
        return load_gam(cls, path)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Sklearn-style score: R² for regression families, accuracy for
        binomial.
        """
        y_arr = to_1d_array(y, name="y")
        if self.family in ("binomial", "bernoulli"):
            p = self.predict(X, scale="response")
            pred_label = (p > 0.5).astype(float)
            return float(np.mean(pred_label == y_arr))
        # Regression: R² = 1 - SS_res / SS_tot
        yhat = self.predict(X, scale="response")
        ss_res = float(np.sum((y_arr - yhat) ** 2))
        y_mean = float(np.mean(y_arr))
        ss_tot = float(np.sum((y_arr - y_mean) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def __getitem__(self, predictors: Union[str, Iterable[str]]) -> "Gam":
        """Subset view — `gam[["x0"]]` returns a view that predicts using
        only the named terms. Other terms' contributions are masked to
        zero before β·lp; use ``scale='deviation'`` to also drop the
        intercept and get the pure marginal effect.

        Pass ``gamrs.Gam.INTERCEPT`` (``"__constant__"``) to include the
        intercept explicitly. Single string is shorthand for a one-element
        iterable.
        """
        import copy
        self._require_fitted()
        if isinstance(predictors, str):
            requested = [predictors]
        else:
            requested = list(predictors)
        known = set(self._effective_predictors or []) | {self.INTERCEPT}
        for name in requested:
            if name not in known:
                raise KeyError(
                    f"unknown predictor {name!r}; known: {sorted(known)}"
                )
        view = copy.copy(self)
        view._subset_mask = set(requested)
        return view

    def __repr__(self) -> str:
        if self._fitted is None:
            return (
                f"Gam(family={self.family!r}, link={self.link!r}, "
                f"predictors={self.predictors!r}) [unfit]"
            )
        return (
            f"Gam(family={self.family!r}, link={self.link!r}, "
            f"predictors={self.predictors!r}, "
            f"k={self._k_used}, scale={self.scale_:.4g}, "
            f"edf_total={self.edf_total_:.4g}, converged={self.converged_})"
        )


# =============================================================================
# Deprecated alias matching v0.x: GAMFitter → Gam with a DeprecationWarning.
# =============================================================================


class GAMFitter(Gam):
    """Deprecated alias of :class:`Gam`. Kept for v0.x drop-in
    compatibility."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "GAMFitter is a deprecated alias of gamrs.Gam; rename to Gam.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


