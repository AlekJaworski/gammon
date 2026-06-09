"""Parity tier — multi-smooth shape-managed families through the Python API.

Validates the multi-smooth `Additive` path for NegBin (profile-θ) and Tweedie
(both profile-p and fixed-p) against mgcv 2-D additive fixtures, at the FFI
boundary the Rust tests don't touch. Also locks the Tweedie p-kwarg toggle:
omit `tweedie_p` → profile-p (p estimated, 2 shape params); pass `tweedie_p`
→ fixed-p (p held, 1 shape param) — the mgcv `tw()` vs `Tweedie(p)` split.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

from conftest import load_fixture, max_rel_err, x_train_2d, y_train

pytestmark = pytest.mark.parity


def _fit_additive_cr(fx, family, **kwargs):
    x = x_train_2d(fx)
    y = y_train(fx)
    k = fx["inputs"]["k"]
    fitted = gamrs.fit_additive(
        family, x, y, [gamrs.CrTerm(0, k=int(k[0])), gamrs.CrTerm(1, k=int(k[1]))], **kwargs
    )
    mu = np.exp(np.asarray(fitted.predict(x)))  # log link -> response
    rel = max_rel_err(mu, np.asarray(fx["mgcv_output"]["predictions_train"]))
    return fitted, rel


def test_additive_nb_parity():
    """NegBin profile-θ, 2-D additive. Rust bar 5e-3; observed ~1.4e-3."""
    fitted, rel = _fit_additive_cr(load_fixture("2d_nb_log_n600_k8_cr"), "nb", theta=5.0)
    assert fitted.converged
    assert len(fitted.rho) == 2
    assert rel < 5e-3, f"NB additive μ rel {rel:.3e} exceeds 5e-3"


def test_additive_scat_parity():
    """scat (scaled-t), 2-D additive, identity link. Rust bar 1.5e-2; ~9e-3.

    scat predicts on the identity scale, so no log-link inverse here. Closes
    the README's "multi-smooth scat reference parity pending" gap at the FFI
    boundary.
    """
    fx = load_fixture("2d_scat_identity_n600_k8_cr")
    x = x_train_2d(fx)
    y = y_train(fx)
    k = fx["inputs"]["k"]
    fitted = gamrs.fit_additive(
        "scat", x, y, [gamrs.CrTerm(0, k=int(k[0])), gamrs.CrTerm(1, k=int(k[1]))]
    )
    assert len(fitted.rho) == 2
    mu = np.asarray(fitted.predict(x))  # identity link → μ directly
    rel = max_rel_err(mu, np.asarray(fx["mgcv_output"]["predictions_train"]))
    assert rel < 2e-2, f"scat additive μ rel {rel:.3e} exceeds 2e-2"


def test_additive_ocat_parity():
    """Multi-smooth ocat vs mgcv ocat(R=4) — the FIRST mgcv ocat parity
    check (parity_ocat.rs was smoke-only). Generated from a proper noisy-
    latent DGP (z = η + logistic) where mgcv AND gamrs both converge cleanly
    (mgcv: 3 iters; gamrs: ~5). On the older noiseless quantile-cut fixtures
    the data is near-separable and mgcv itself blows the latent scale up
    (θ≈181) or crashes — gamrs's θ∈(−3,3) bound is more robust there. This
    pins the well-posed regime: converged_=True and predict_proba agreement.
    """
    import warnings

    fx = load_fixture("2d_ocat_r4_n1500_k8_cr")
    x = x_train_2d(fx)
    y = y_train(fx)
    k = fx["inputs"]["k"]
    R = int(fx["inputs"]["r"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        g = gamrs.Gam(
            family="ocat", r=R,
            terms=[gamrs.CrTerm(0, k=int(k[0])), gamrs.CrTerm(1, k=int(k[1]))],
        ).fit(x, y)
    assert g.converged_, f"multi-smooth ocat should converge on well-posed data (conv={g.converged_})"
    proba = np.asarray(g.predict_proba(x))
    mgcv_proba = np.asarray(fx["mgcv_output"]["proba"])
    assert proba.shape == mgcv_proba.shape == (len(y), R)
    max_abs = float(np.abs(proba - mgcv_proba).max())
    mean_abs = float(np.abs(proba - mgcv_proba).mean())
    agree = float((np.argmax(proba, axis=1) == np.argmax(mgcv_proba, axis=1)).mean())
    print(f"\n[ocat parity] max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} class_agree={agree:.3f}")
    # Observed: mean_abs ~1.8e-3, max_abs ~1.9e-2, class_agree ~0.983.
    assert agree > 0.97, f"gamrs/mgcv ocat class agreement {agree:.3f} too low"
    assert mean_abs < 5e-3, f"ocat predict_proba mean abs diff {mean_abs:.3e} exceeds 5e-3"
    assert max_abs < 5e-2, f"ocat predict_proba max abs diff {max_abs:.3e} exceeds 5e-2"


def test_additive_tweedie_profile_p_parity():
    """Tweedie profile-p (no tweedie_p): p estimated → 2 shape params."""
    fitted, rel = _fit_additive_cr(load_fixture("2d_tw_profile_log_n600_k8_cr"), "tw")
    assert len(fitted.rho) == 2
    assert np.asarray(fitted.shape_params).size == 2, "profile-p must keep [logφ, p_transform]"
    assert rel < 1.5e-2, f"Tweedie profile-p additive μ rel {rel:.3e} exceeds 1.5e-2"


def test_additive_tweedie_fixed_p_parity():
    """Tweedie fixed-p (tweedie_p=1.5): p held → 1 shape param [logφ]."""
    fitted, rel = _fit_additive_cr(
        load_fixture("2d_tw_fixed_p15_log_n600_k8_cr"), "tw", tweedie_p=1.5
    )
    assert len(fitted.rho) == 2
    assert np.asarray(fitted.shape_params).size == 1, "fixed-p must drop the p shape axis"
    assert rel < 1.5e-2, f"Tweedie fixed-p additive μ rel {rel:.3e} exceeds 1.5e-2"


@pytest.mark.smoke
def test_tweedie_p_kwarg_toggles_mode():
    """The p-kwarg toggle (mgcv_rust convention): None → profile, val → fixed.

    Fixed-p must hold p across inits; profile-p must estimate the same p
    regardless of the seed value. Single-smooth, fast — also runs in smoke.
    """
    fx = load_fixture("2d_tw_profile_log_n600_k8_cr")
    x = x_train_2d(fx)
    y = y_train(fx)

    def p_from(fitted):
        sp = np.asarray(fitted.shape_params)
        return None if sp.size == 1 else 1.0 + 1.0 / (1.0 + np.exp(-sp[-1]))

    # profile-p: omit tweedie_p -> 2 shape params, p estimated.
    prof = gamrs.GAM("tw", k=10).fit(x[:, 0], y)
    assert np.asarray(prof._fitted.shape_params).size == 2
    assert p_from(prof._fitted) is not None

    # fixed-p: pass tweedie_p -> 1 shape param, p NOT estimated.
    for val in (1.3, 1.5, 1.7):
        fix = gamrs.GAM("tw", k=10).fit(x[:, 0], y, tweedie_p=val)
        assert np.asarray(fix._fitted.shape_params).size == 1, "fixed-p drops the p axis"
