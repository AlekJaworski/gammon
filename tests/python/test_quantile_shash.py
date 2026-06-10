"""Focused tests for the SHASH err-param σ path in quantile ``fast_oos``.

These lock the port of mgcv_rust's qgam ``.getErrParam`` (SHASH-distribution
bandwidth → ELF scale ``co``) into gamrs's ``fit_quantile(preset="fast_oos")``.

Coverage:

- :mod:`gamrs._shash` correctness: ``err`` ∈ (0, 1]; degenerate-residual
  fallback raises (so the caller drops to the documented ``err = 0.05``).
- ``fast_oos`` exposes ``sigma_ == co_`` (the σ = co = co_auto mapping) and a
  finite coverage shift.
- The SHASH σ fixes the extreme tail: on a heteroscedastic 1-D dataset the
  τ = 0.99 OOS pinball is materially better than the bare ``elf_sigma = 0``
  Rust heuristic, and (when mgcv_rust is importable) within 5% of mgcv_rust's
  own ``fast_oos`` at τ ∈ {0.9, 0.95, 0.99}.
"""

from __future__ import annotations

import numpy as np
import pytest

# `gamrs._shash` imports scipy at module load, but scipy is an *optional*
# dependency (pyproject `[project.optional-dependencies] quantile`) — absent on
# platforms without a scipy wheel and not installed in the base CI env. Skip
# the whole module cleanly when scipy is unavailable rather than erroring at
# collection time (which aborts the entire pytest run).
pytest.importorskip("scipy")

import gamrs
from gamrs._shash import compute_err_param, fit_shash

pytestmark = pytest.mark.smoke

SEED = 20260529


def _hetero_split(n_tr: int = 2000, n_te: int = 1000, seed: int = SEED):
    """x ~ U(0,1); y = sin(2πx) + (0.2 + 0.6x)·N(0,1) — heteroscedastic."""
    rng = np.random.default_rng(seed)
    n = n_tr + n_te
    x = rng.uniform(0.0, 1.0, size=n)
    noise = rng.standard_normal(n)
    y = np.sin(2.0 * np.pi * x) + (0.2 + 0.6 * x) * noise
    return x[:n_tr, None], y[:n_tr], x[n_tr:, None], y[n_tr:]


def _pinball(y, yhat, tau):
    r = np.asarray(y) - np.asarray(yhat)
    return float(np.maximum(tau * r, (tau - 1.0) * r).mean())


# --------------------------------------------------------------------------- #
# SHASH helper correctness                                                     #
# --------------------------------------------------------------------------- #


def test_compute_err_param_in_unit_range():
    rng = np.random.default_rng(SEED)
    r = rng.standard_normal(2000)
    errs = compute_err_param(r, d=6.0, qu_list=[0.5, 0.9, 0.95, 0.99])
    assert len(errs) == 4
    assert all(np.isfinite(e) and 0.0 < e <= 1.0 for e in errs)


def test_fit_shash_degenerate_raises():
    # Zero-variance residuals must raise so the caller falls back to err=0.05.
    with pytest.raises(Exception):
        fit_shash(np.zeros(64))


# --------------------------------------------------------------------------- #
# fast_oos attributes + σ = co = co_auto mapping                              #
# --------------------------------------------------------------------------- #


def test_fast_oos_exposes_shash_sigma_and_co():
    x_tr, y_tr, _, _ = _hetero_split(n_tr=800, n_te=1, seed=SEED)
    g = gamrs.fit_quantile(x_tr, y_tr, 0.95, k=10, preset="fast_oos")
    # SHASH path => co_ set, σ = co (mgcv_rust mapping).
    assert g.co_ is not None and g.co_ > 0.0
    assert g.sigma_ == pytest.approx(g.co_, rel=1e-12)
    assert g.coverage_shift_ is not None and np.isfinite(g.coverage_shift_)
    assert g.tune_info_ is None  # no CV ran


def test_explicit_sigma_skips_shash():
    x_tr, y_tr, _, _ = _hetero_split(n_tr=800, n_te=1, seed=SEED)
    g = gamrs.fit_quantile(x_tr, y_tr, 0.9, k=10, sigma=0.07)
    assert g.co_ is None  # SHASH not used
    assert g.coverage_shift_ is None  # no coverage calibration


# --------------------------------------------------------------------------- #
# Tail-quality gate                                                            #
# --------------------------------------------------------------------------- #


def test_fast_oos_beats_bare_heuristic_at_tail():
    """SHASH σ must materially beat the bare ``elf_sigma=0`` heuristic at τ=0.99."""
    x_tr, y_tr, x_te, y_te = _hetero_split()
    tau = 0.99

    # SHASH fast_oos.
    g = gamrs.fit_quantile(x_tr, y_tr, tau, k=10, preset="fast_oos")
    pb_shash = _pinball(y_te, np.asarray(g.predict(x_te)).ravel(), tau)

    # Bare heuristic baseline: σ=0 + the same coverage calibration, no SHASH.
    g0 = gamrs.fit_quantile(x_tr, y_tr, tau, k=10, sigma=0.0, coverage_calibrate=True)
    pb_bare = _pinball(y_te, np.asarray(g0.predict(x_te)).ravel(), tau)

    # SHASH must cut the tail pinball by a clear margin (historically 0.0181
    # -> ~0.0134, ~26%); require at least a 5% improvement to lock the win.
    assert pb_shash < 0.95 * pb_bare, (pb_shash, pb_bare)
    # And land comfortably below the documented ~0.0157 mgcv_rust target.
    assert pb_shash < 0.017, pb_shash


def test_fast_oos_within_5pct_of_mgcv_rust():
    """When mgcv_rust is importable, gamrs fast_oos ties it within 5% at all τ."""
    mgcv_rust = pytest.importorskip("mgcv_rust")
    x_tr, y_tr, x_te, y_te = _hetero_split()
    for tau in (0.9, 0.95, 0.99):
        g = gamrs.fit_quantile(x_tr, y_tr, tau, k=10, preset="fast_oos")
        pb_g = _pinball(y_te, np.asarray(g.predict(x_te)).ravel(), tau)

        gm = mgcv_rust.fit_quantile(x_tr, y_tr, tau, k=[10], preset="fast_oos")
        gm = gm[0] if isinstance(gm, tuple) else gm
        pb_m = _pinball(y_te, np.asarray(gm.predict(x_te)).ravel(), tau)

        # Within ~5% above mgcv_rust (better is fine — gamrs may beat it).
        assert pb_g <= 1.05 * pb_m, (tau, pb_g, pb_m)


def test_lss_shash_fixes_skewed_tails():
    """`fit_quantile_lss(shape="shash")` corrects the tail mis-coverage that
    the Gaussian shape suffers on skewed data.

    On strongly right-skewed (centered-lognormal) conditional noise, the
    Gaussian shape assumes symmetry and badly mis-covers the tails — most
    dramatically the *lower* tail, where q̂₀.₁ can fall below nearly all the
    data. SHASH fits the skew on the standardised residuals and corrects it.
    This locks the value of the optional `shape="shash"` path.
    """
    rng = np.random.default_rng(7)
    n = 1500
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    mu = np.sin(2 * np.pi * x0) + 1.5 * (x1 - 0.5)
    scale = 0.2 + 0.3 * x0
    raw = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    noise = (raw - np.exp(0.8**2 / 2)) / np.std(raw)  # mean 0, var 1, right-skewed
    y = mu + scale * noise
    X = np.column_stack([x0, x1])
    terms = [gamrs.CrTerm(0, k=10), gamrs.CrTerm(1, k=8)]

    fg = gamrs.fit_quantile_lss(X, y, terms=terms, shape="gaussian")
    fs = gamrs.fit_quantile_lss(X, y, terms=terms, shape="shash")

    tails = [0.05, 0.1, 0.9, 0.95, 0.99]

    def cov_err(fit):
        return {
            t: abs(float((y <= np.asarray(fit.predict_quantile(X, t)).ravel()).mean()) - t)
            for t in tails
        }

    eg, es = cov_err(fg), cov_err(fs)

    # SHASH actually fit non-trivial skew (eps != 0).
    assert fs.shash_params_ is not None and abs(fs.shash_params_[2]) > 0.2, fs.shash_params_
    # The Gaussian shape badly mis-covers the lower tail of right-skewed data;
    # SHASH brings it back near target.
    assert eg[0.1] > 0.05, f"expected Gaussian lower-tail mis-coverage, got err={eg[0.1]:.3f}"
    assert es[0.1] < 0.04, f"SHASH lower-tail coverage err {es[0.1]:.3f} too large"
    # And SHASH is better on average across the extreme tails.
    assert np.mean(list(es.values())) < np.mean(list(eg.values())), (es, eg)
