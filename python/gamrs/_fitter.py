"""Sklearn-style ``Gam`` wrapper — drop-in for ``mgcv_rust.Gam``.

Mirrors the v0.x ``mgcv_rust.Gam`` API surface 1-1 for the basic
single-smooth use case. See :mod:`gamrs` for the full migration story
and the list of features that raise ``NotImplementedError`` (with a
pointer back at v0.x for the ones not yet wired).
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

from . import _gamrs_native
from ._coerce import FAMILY_TO_GAMRS, to_1d_array, to_2d_with_columns
from ._low_level import CrTerm, CrStableTerm, ReTerm, TeTerm, Term, _term_to_tuple
from ._stubs import GamSummary, TermContributions  # noqa: F401

ArrayLike = Any  # avoid hard dep on pandas/polars typing


# =============================================================================
# Gam — sklearn-style wrapper. Mirrors mgcv_rust.Gam.
# =============================================================================


class Gam:
    """Sklearn-style gamrs GAM wrapper — drop-in replacement for
    :class:`mgcv_rust.Gam` (single-smooth scope today)."""

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

        if self.auto_k:
            warnings.warn(
                "auto_k=True is not yet wired in gamrs; using a single fit with "
                "k=k_default (or term_k_mapping). Drop auto_k to silence.",
                UserWarning,
                stacklevel=2,
            )

        # Forward-compat: stash unknown kwargs without erroring so a
        # textual mgcv_rust → gamrs substitution doesn't blow up at
        # construction time.
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
        return out

    # ------------------------ fit / predict --------------------------- #

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Any = None) -> "Gam":
        """Fit the GAM. Drop-in for ``mgcv_rust.Gam.fit``.

        ``X`` may be a DataFrame / 2-D ndarray / 1-D ndarray. Multi-column
        X dispatches to the additive multi-smooth path (one CR term per
        column by default; ``predictor_basis_map`` lets you switch a
        column to ``"re"``). For the typed-term API, pass ``terms=`` to
        the constructor — that path bypasses predictor-name resolution.
        """
        X_arr, cols = to_2d_with_columns(X, self.predictors)
        y_arr = to_1d_array(y, name="y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]} elements"
            )

        if self.predictors is None:
            self.predictors = cols
        self._effective_predictors = list(cols)
        self._original_predictors = list(cols)
        self.dropped_predictors_ = {}
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
        family_kw = self._family_kwargs_for_native()

        # Build the term list — either from self.terms (typed API) or
        # derived from columns + predictor_basis_map (v0.x API).
        if self.terms is not None:
            term_objs = list(self.terms)
            self._k_used = [
                t.k if isinstance(t, (CrTerm, CrStableTerm)) else
                (t.k[0] * t.k[1] if isinstance(t, TeTerm) else 0)
                for t in term_objs
            ]
        else:
            term_objs = self._build_terms_from_columns(X_arr, cols)
            self._k_used = [
                t.k if isinstance(t, (CrTerm, CrStableTerm)) else 0
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
            return self

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
        return self

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
                raise NotImplementedError(
                    f"predictor_basis_map[{pname!r}]={bs_override!r} (parametric / "
                    "linear unsmoothed term) is not yet wired in gamrs; use the "
                    "v0.x mgcv_rust.Gam wrapper for parametric columns."
                )
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
        if scale == "deviation":
            raise ValueError(
                "scale='deviation' is only meaningful on subset views; "
                "gamrs doesn't yet support subset views — use scale='link'."
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

        x = self._coerce_predict_X(X)
        # gamrs's native predict_ci returns (mean, lo, hi) on link OR response.
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
        """v0.x's auto-k trace. gamrs doesn't run auto-k yet, so empty."""
        return []

    @property
    def ocat_theta_(self) -> np.ndarray:
        """v0.x: converged log-gap thresholds for ocat. gamrs doesn't yet
        expose these through the bindings — returns an empty array."""
        if self.family != "ocat":
            raise AttributeError("ocat_theta_ is only available for family='ocat'")
        return np.array([])

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
        raise NotImplementedError(
            "get_posterior_samples() is not yet wired in gamrs. Use "
            "mgcv_rust.Gam for posterior sampling."
        )

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return ``(n, 2)`` probability matrix for binomial; ``(n, R)``
        for ocat. gamrs currently supports binomial only — ocat raises.
        """
        if self.family not in ("binomial", "bernoulli"):
            raise NotImplementedError(
                f"predict_proba() is currently only wired for "
                f"family='binomial'/'bernoulli'; got family={self.family!r}."
            )
        p = self.predict(X, scale="response")
        return np.column_stack([1.0 - p, p])

    def partial_effect(
        self,
        predictor: str,
        grid_n: int = 100,
        level: Optional[float] = 0.95,
    ) -> Any:
        raise NotImplementedError(
            "partial_effect() is not yet wired in gamrs. Use "
            "mgcv_rust.Gam.partial_effect() for the per-smooth plot data."
        )

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "plot() is not yet wired in gamrs. Use mgcv_rust.Gam.plot() or "
            "build a matplotlib figure from partial_effect data."
        )

    def summary(self) -> GamSummary:
        f = self._require_fitted()
        return GamSummary(
            family=self.family,
            link=self.link,
            n=int(f.n),
            intercept=float(f.beta[0]),
            scale=float(f.scale),
            edf_total=float(f.edf_total),
            lambda_=self.lambda_,
            converged=bool(f.converged),
            n_iters=int(f.n_iters),
        )

    # ----------------------- Persistence (save/load) ----------------- #
    # Bulk of the on-disk format lives in `_persistence.py`.

    def serialize(self) -> bytes:
        """Native bytes (MAGIC | VERSION | LEN | JSON). Pair with
        :meth:`deserialize` or :meth:`save` / :meth:`load`."""
        return bytes(self._require_fitted().serialize())

    @classmethod
    def deserialize(cls, payload: bytes) -> "Gam":
        """Rebuild from native bytes only — wrapper-side metadata is
        defaulted. Use :meth:`Gam.load` for the metadata-aware path."""
        gam = cls.__new__(cls)  # bypass __init__; defaults below
        gam._fitted = _gamrs_native.FittedGam.deserialize(payload)
        gam.__dict__.update(
            predictors=None, _effective_predictors=None,
            _original_predictors=None, dropped_predictors_={},
            X=None, y=None, sample_weight=None, _k_used=None,
            family="gaussian", link="identity", _gamrs_family="gaussian",
            method="REML", target="y", design="cr",
            term_k_mapping={}, term_pc_mapping={}, predictor_basis_map={},
            consider_categorical=False, auto_k=False, discrete=False,
            df=None, tweedie_p=None, negbin_theta=None, r=None,
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


