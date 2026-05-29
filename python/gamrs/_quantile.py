"""qgam-style σ-calibration helpers for the ELF / quantile family.

Why this lives in Python: the inner GAM fit is fast Rust (the ELF family
runs through gamrs's standard PIRLS + REML); only the outer σ search is
cheap Python loops. This module ports v0.x's
:func:`mgcv_rust.fit_quantile` and :func:`mgcv_rust.tune_quantile_sigma`
to gamrs.

Two helpers:

- :func:`tune_quantile_sigma`: K-fold CV with Brent on log σ. Returns σ̂
  and a small info dict (loss curve etc.). Pinball loss is the only
  criterion supported here — the bootstrap-KL ("cal_kl") variant from
  v0.x is deferrable to a follow-up; the location-scale "lss" pipeline
  is also deferrable.

- :func:`fit_quantile`: one-shot helper that auto-calibrates σ then fits
  a :class:`gamrs.Gam` at the chosen σ. Returns the fitted ``Gam`` with
  a ``sigma_`` attribute attached.

Why CV-on-σ and not LAML / REML: Fasiolo et al. 2021 show ELF's
likelihood is a Gibbs posterior, not a true likelihood — the MLE σ is
structurally degenerate (the REML score collapses to σ→0 because
in-sample pinball shrinks faster than complexity penalties grow). CV is
the only in-sample-free criterion that breaks the tie cleanly.

Deferred (TODO follow-up):

- ``fit_quantile_lss`` / ``QuantileLSSFit`` (heteroskedastic location-
  scale port of qgam ≥1.3).
- ``loss="cal_kl"`` bootstrap-KL criterion (qgam's ``tuneLearnFast``
  ``loss="cal"`` path).
- v0.x's auto-σ via ``compute_err_param`` (qgam's ``.getErrParam``)
  initial heuristic (depends on the SHASH helper, also unported).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from . import _gamrs_native
from ._coerce import to_1d_array, to_2d_with_columns
from ._fitter import Gam


def _pinball(y: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Mean pinball loss at quantile ``tau``."""
    r = y - y_pred
    return float(np.maximum(tau * r, (tau - 1.0) * r).mean())


def _build_folds(
    n: int, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K equal-size folds via a permuted index split.

    Returns a list of ``(test_idx, train_idx)`` pairs.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // n_folds
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        test_idx = perm[i * fold_size : (i + 1) * fold_size]
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        folds.append((test_idx, np.where(train_mask)[0]))
    return folds


def _fit_elf_native(
    x_2d: np.ndarray,
    y: np.ndarray,
    tau: float,
    sigma: float,
    k: int,
    design: str,
) -> Any:
    """Run a single ELF fit at the given (τ, σ) directly through the
    native binding. Returns the ``FittedGam`` object.

    The σ argument is plumbed as the native binding's ``elf_sigma`` kwarg
    (gamrs's name for qgam's σ scale parameter). ``sigma=0.0`` defers to
    the Rust-side heuristic, matching v0.x semantics.
    """
    return _gamrs_native.fit(
        "elf",
        x_2d,
        y,
        k=int(k),
        design=design,
        tau=float(tau),
        elf_sigma=float(sigma),
    )


def _calibrate_quantile_intercept(
    fitted: Any, x_2d: np.ndarray, y: np.ndarray, tau: float
) -> float:
    """qgam-style coverage calibration (policy; ported from mgcv_rust).

    After the ELF fit, shift the intercept so the empirical training coverage
    matches τ: ``shift = τ-th order statistic of (y − μ̂)`` (floor-index, the
    exact convention of mgcv_rust's native ``calibrate_quantile_intercept``),
    then apply it via the core's generic :meth:`shift_intercept` primitive.
    Cheap (one predict + one quantile) and family-agnostic mechanism — the
    quantile-specific policy lives here, not in the Rust core.

    Returns the applied shift.
    """
    # ELF uses an identity link, so the link-scale prediction IS μ̂.
    mu = np.asarray(fitted.predict(x_2d), dtype=np.float64).ravel()
    resid = np.asarray(y, dtype=np.float64).ravel() - mu
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        raise ValueError("cannot calibrate quantile intercept from non-finite residuals")
    resid.sort()
    idx = int(np.clip(np.floor((resid.size - 1) * tau), 0, resid.size - 1))
    shift = float(resid[idx])
    fitted.shift_intercept(shift)
    return shift


def _cv_loss_at_sigma(
    sigma: float,
    x_2d: np.ndarray,
    y: np.ndarray,
    tau: float,
    k: int,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    design: str,
) -> float:
    """K-fold mean pinball loss at the given σ. Folds that fail to fit
    or produce NaN predictions contribute ``+inf``.
    """
    losses: list[float] = []
    for test_idx, train_idx in folds:
        try:
            x_tr = np.ascontiguousarray(x_2d[train_idx], dtype=np.float64)
            y_tr = np.ascontiguousarray(y[train_idx], dtype=np.float64)
            x_te = np.ascontiguousarray(x_2d[test_idx], dtype=np.float64)
            f = _fit_elf_native(x_tr, y_tr, tau=tau, sigma=sigma, k=k, design=design)
            y_pred = np.asarray(f.predict(x_te))
            if not np.all(np.isfinite(y_pred)):
                losses.append(np.inf)
                continue
            losses.append(_pinball(y[test_idx], y_pred, tau))
        except Exception:
            losses.append(np.inf)
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def tune_quantile_sigma(
    X: Any,
    y: Any,
    tau: float,
    k: int = 10,
    K_folds: int = 5,
    brent_bracket: Optional[tuple[float, float]] = None,
    design: str = "cr",
    seed: int = 0,
    xatol: float = 0.05,
) -> tuple[float, dict[str, Any]]:
    """Pick σ for the ELF / quantile family via K-fold pinball CV.

    Runs SciPy's bounded Brent on ``log σ`` and minimises mean held-out
    pinball loss across ``K_folds`` folds. Returns ``(σ̂, info)`` where
    ``info`` contains the loss at the optimum, the Brent eval count, and
    the (log) bracket.

    Args:
      X: ``(n, d)`` design matrix (DataFrame / ndarray / 1-D vector
        accepted).
      y: ``(n,)`` response.
      tau: target quantile, in ``(0, 1)``.
      k: basis dimension for the single ELF smooth (matches the inner
        ``Gam(family='elf')`` fit).
      K_folds: number of CV folds (default 5).
      brent_bracket: ``(log_lo, log_hi)`` bracket on ``log σ``. Defaults
        to ``(log(0.05 · sd(y)), log(5 · sd(y)))`` — a 100× span centred
        on the response scale, matching v0.x's default.
      design: spline basis for the smooth (``'cr'`` default).
      seed: RNG seed for fold construction.
      xatol: Brent absolute tolerance in ``log σ`` space (default 0.05 ≈
        5% in σ — empirically enough for stable σ̂ on n ≥ 200).
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as exc:
        raise ImportError(
            "tune_quantile_sigma requires scipy. Install with `pip install scipy`."
        ) from exc

    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1); got tau={tau}")

    x_2d, _ = to_2d_with_columns(X, None)
    y_arr = to_1d_array(y, name="y")
    if x_2d.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {x_2d.shape[0]} rows but y has {y_arr.shape[0]} elements"
        )
    n = y_arr.shape[0]

    folds = _build_folds(n, K_folds, seed)

    if brent_bracket is None:
        y_scale = float(np.std(y_arr))
        if y_scale <= 0.0:
            y_scale = 1.0
        brent_bracket = (np.log(0.05 * y_scale), np.log(5.0 * y_scale))
    lo, hi = float(brent_bracket[0]), float(brent_bracket[1])
    if not lo < hi:
        raise ValueError(
            f"brent_bracket must satisfy lo < hi; got ({lo}, {hi})"
        )

    def objective(log_sigma: float) -> float:
        return _cv_loss_at_sigma(
            float(np.exp(log_sigma)), x_2d, y_arr, tau, k, folds, design
        )

    result = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": float(xatol)},
    )
    sigma_hat = float(np.exp(result.x))
    info: dict[str, Any] = {
        "sigma": sigma_hat,
        "log_sigma": float(result.x),
        "cv_loss": float(result.fun),
        "n_brent_evals": int(result.nfev),
        "n_folds": int(K_folds),
        "bracket_log_sigma": (lo, hi),
        "tau": float(tau),
        "k": int(k),
    }
    return sigma_hat, info


def fit_quantile(
    X: Any,
    y: Any,
    tau: float,
    k: int = 10,
    K_folds: int = 5,
    brent_bracket: Optional[tuple[float, float]] = None,
    design: str = "cr",
    seed: int = 0,
    xatol: float = 0.05,
    sigma: Optional[float] = None,
    coverage_calibrate: bool = False,
    preset: Optional[str] = None,
) -> Gam:
    """Fit a quantile (ELF) GAM with σ chosen by K-fold pinball CV.

    This is the v0.x ``mgcv_rust.fit_quantile`` convenience wrapper,
    ported to gamrs. The inner ELF fit goes through
    :class:`gamrs.Gam` with ``family='quantile'`` (which dispatches to
    the Rust ELF family — see :data:`gamrs._coerce.FAMILY_TO_GAMRS`); the
    outer σ search is :func:`tune_quantile_sigma`.

    Args:
      X: ``(n, d)`` design (DataFrame / ndarray / 1-D vector accepted).
      y: ``(n,)`` response.
      tau: target quantile in ``(0, 1)``.
      k: basis dim for the single ELF smooth.
      K_folds: CV folds for σ tuning.
      brent_bracket: ``(log_lo, log_hi)`` bracket on ``log σ``.
      design: basis kind (default ``'cr'``).
      seed: RNG seed for fold construction.
      xatol: Brent tolerance on ``log σ`` (default 0.05).
      sigma: if given, skip CV and fit at this σ directly (escape hatch
        for callers that already have a tuned σ̂).
      coverage_calibrate: if True, after the fit shift the intercept so the
        empirical training coverage matches τ (qgam-style coverage
        calibration, ported from mgcv_rust). Cheap post-fit step.
      preset: convenience bundles ported from mgcv_rust's OOS quantile paths:

        - ``"fast_oos"``: heuristic σ (native, NO CV) + coverage calibration.
          The speed/quality balance — fits in one ELF pass; matches mgcv_rust
          OOS pinball at a fraction of the CV cost.
        - ``"quality_oos"``: CV-tuned σ + coverage calibration.

    Returns:
      A fitted :class:`gamrs.Gam` with extra attributes attached:

      - ``sigma_``: the σ used in the final fit (``0.0`` = native heuristic).
      - ``tune_info_``: the :func:`tune_quantile_sigma` info dict
        (``None`` when σ was not CV-tuned).
      - ``coverage_shift_``: the applied coverage-calibration shift, or
        ``None`` when ``coverage_calibrate`` was off.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1); got tau={tau}")

    # Preset resolution mirrors mgcv_rust._quantile.fit_quantile.
    if preset is not None:
        if preset == "fast_oos":
            coverage_calibrate = True
            if sigma is None:
                sigma = 0.0  # native heuristic σ — skips the CV path below
        elif preset == "quality_oos":
            coverage_calibrate = True  # σ stays CV-tuned (sigma is None)
        else:
            raise ValueError(
                f"unknown quantile preset {preset!r}; expected 'fast_oos' or 'quality_oos'"
            )

    info: Optional[dict[str, Any]] = None
    if sigma is None:
        sigma, info = tune_quantile_sigma(
            X,
            y,
            tau=tau,
            k=k,
            K_folds=K_folds,
            brent_bracket=brent_bracket,
            design=design,
            seed=seed,
            xatol=xatol,
        )

    # Build + fit the Gam ONCE at the target (τ, σ). We plumb τ/σ through the
    # Gam's ELF config so `g.fit()` lands at the right fit directly — no
    # fit-then-replace double pass (matches mgcv_rust's single fit). The
    # `Gam.fit` path sets `_gamrs_family='elf'` for family='quantile'.
    g = Gam(family="quantile", k_default=int(k), design=design)
    g._elf_tau = float(tau)  # type: ignore[attr-defined]
    g._elf_sigma = float(sigma)  # type: ignore[attr-defined]
    g.fit(X, y)

    # qgam-style coverage calibration (policy lives in this module; the
    # intercept shift is applied via the core's generic shift_intercept).
    coverage_shift: Optional[float] = None
    if coverage_calibrate:
        x_2d_full, _ = to_2d_with_columns(X, None)
        y_full = to_1d_array(y, name="y")
        coverage_shift = _calibrate_quantile_intercept(
            g._fitted,
            np.ascontiguousarray(x_2d_full, dtype=np.float64),
            np.ascontiguousarray(y_full, dtype=np.float64),
            tau,
        )

    # v0.x parity: expose σ̂ + the calibration trace on the fitted Gam.
    g.sigma_ = float(sigma)  # type: ignore[attr-defined]
    g.tau_ = float(tau)  # type: ignore[attr-defined]
    g.tune_info_ = info  # type: ignore[attr-defined]
    g.coverage_shift_ = coverage_shift  # type: ignore[attr-defined]
    return g
