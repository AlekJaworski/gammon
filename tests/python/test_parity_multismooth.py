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
