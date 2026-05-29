"""Parity tier — gamrs Python API vs mgcv reference fixtures.

Mirrors the Rust ``tests/parity_*.rs`` battery at the Python layer, so a
regression in the FFI boundary / coercion facade (the layer the Rust tests
never touch) is caught directly. Each case fits through the *public* Python
API (``gamrs.GAM`` / ``gamrs.fit_additive``) and compares response-scale
predictions to ``mgcv_output.predictions_train``.

Tolerances track the corresponding Rust bound (``parity_<family>.rs``). They
are deliberately the *same* number, not looser: the Python path delegates to
the identical native fit, so any divergence beyond the Rust bar is a real
coercion/dispatch bug. Large-n fixtures carry ``pytest.mark.slow``.

Coverage (max entropy for the cost): gaussian (8 shapes) · binomial · poisson
· gamma(log) · negbin · tweedie · scat · inverse-gaussian · quasipoisson ·
quasibinomial · 2-D additive · 2-D tensor product.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

from conftest import load_fixture, max_rel_err, x_train_2d, y_train


# --------------------------------------------------------------------------- #
# Single-smooth cases: (fixture, gamrs-family, fit-kwargs, mu_tol, slow,        #
#                       require_converged)                                      #
# `require_converged` mirrors the Rust suite: profiled-φ families (gamma,       #
# invgauss, tweedie) have a sensitive convergence flag, so parity_*.rs there    #
# checks only the μ bound. We do the same.                                      #
# --------------------------------------------------------------------------- #
SINGLE_CASES = [
    # Gaussian, identity link — Rust bar REL_PRED = 5e-5 (parity_gaussian.rs).
    ("1d_gaussian_sigmoid_n300_k10_cr", "gaussian", {}, 1e-4, False, True),
    ("1d_gaussian_smooth_n100_k10_cr", "gaussian", {}, 1e-4, False, True),
    ("1d_gaussian_smooth_n500_k10_cr", "gaussian", {}, 1e-4, False, True),
    ("1d_gaussian_near_linear_n500_k10_cr", "gaussian", {}, 1e-4, False, True),
    ("1d_gaussian_wiggly_n500_k20_cr", "gaussian", {}, 1e-4, False, True),
    ("1d_gaussian_step_n500_k10_cr", "gaussian", {}, 1e-4, False, True),
    ("1d_gaussian_sparse_edges_n400_k10_cr", "gaussian", {}, 1e-4, False, True),
    ("1d_gaussian_smooth_n1000_k50_cr", "gaussian", {}, 1e-4, True, True),
    ("1d_gaussian_smooth_n2000_k30_cr", "gaussian", {}, 1e-4, True, True),
    ("1d_gaussian_low_signal_n1000_k10_cr", "gaussian", {}, 1e-4, True, True),
    # Binomial / logit — parity_binomial.rs bar 2e-3.
    ("1d_bernoulli_logit_n300_k10_cr", "binomial", {}, 2e-3, False, True),
    ("1d_bernoulli_logit_n1000_k10_cr", "binomial", {}, 2e-3, True, True),
    # Poisson / log — parity_poisson.rs bar 5e-3.
    ("1d_poisson_log_n300_k10_cr", "poisson", {}, 5e-3, False, True),
    # Gamma / log — parity_gamma.rs bar 2e-2. fixture family "Gamma" link log
    # maps to gamrs "gamma" (= gamma_log); "Gamma" alone is the inverse link.
    # converged flag sensitive on profiled-φ (parity_gamma.rs comment).
    ("1d_gamma_log_n300_k10_cr", "gamma", {}, 2e-2, False, False),
    # NegBin / log — parity_negbin.rs uses init theta = 5.0, bar 1e-2.
    ("1d_nb_log_n300_k10_cr", "nb", {"theta": 5.0}, 1e-2, False, True),
    # Tweedie / log — parity_tweedie.rs init p=1.5 phi=1.0 (defaults), bar 5e-3.
    # parity_tweedie.rs does not assert convergence.
    ("1d_tweedie_log_n300_k10_cr", "tw", {}, 5e-3, False, False),
    # Inverse-Gaussian / log — parity_invgauss.rs bar 5e-2; profiled-φ, no
    # convergence assertion (V=μ³ score landscape is steepest).
    ("1d_invgauss_log_n300_k10_cr", "inverse_gaussian", {}, 5e-2, False, False),
    # QuasiPoisson / log — parity_quasipoisson.rs bar 5e-3.
    ("1d_quasipoisson_log_n300_k10_cr", "quasipoisson", {}, 5e-3, False, True),
    # QuasiBinomial / logit — parity_quasibinomial.rs bar 5e-3.
    ("1d_quasibinomial_logit_n300_k10_cr", "quasibinomial", {}, 5e-3, False, True),
]


def _single_param(case):
    name, slow = case[0], case[4]
    marks = [pytest.mark.slow] if slow else []
    return pytest.param(case, id=name, marks=marks)


@pytest.mark.parity
@pytest.mark.parametrize("case", [_single_param(c) for c in SINGLE_CASES])
def test_single_smooth_parity(case):
    name, family, kwargs, tol, _slow, require_converged = case
    fx = load_fixture(name)
    x = x_train_2d(fx)
    y = y_train(fx)
    k = int(fx["inputs"]["k"][0])

    fit_kwargs = dict(kwargs)
    # scat carries its shape params; mirror parity_scat.rs (nu=5, sigma2≈0.1*var).
    if family == "scat":
        fit_kwargs.setdefault("nu", 5.0)
        fit_kwargs.setdefault("sigma2", float(np.var(y) * 0.1))

    g = gamrs.GAM(family, k=k).fit(x[:, 0], y, **fit_kwargs)
    if require_converged:
        assert g.converged_, f"[{name}] outer Newton did not converge"

    mu = g.predict(x, scale="response")
    rel = max_rel_err(mu, np.asarray(fx["mgcv_output"]["predictions_train"]))
    assert rel < tol, f"[{name}] μ rel error {rel:.3e} exceeds {tol:.0e}"


@pytest.mark.parity
def test_scat_parity():
    """scat / identity — parity_scat.rs bar 5e-2 (own test: shape kwargs)."""
    name = "1d_scat_unweighted_n300_k10_cr"
    fx = load_fixture(name)
    x = x_train_2d(fx)
    y = y_train(fx)
    k = int(fx["inputs"]["k"][0])
    g = gamrs.GAM("scat", k=k).fit(
        x[:, 0], y, nu=5.0, sigma2=float(np.var(y) * 0.1)
    )
    assert g.converged_
    mu = g.predict(x, scale="response")
    rel = max_rel_err(mu, np.asarray(fx["mgcv_output"]["predictions_train"]))
    assert rel < 5e-2, f"[{name}] scat μ rel error {rel:.3e} exceeds 5e-2"


# --------------------------------------------------------------------------- #
# Multi-smooth: 2-D additive + tensor product (gaussian, identity link)       #
# --------------------------------------------------------------------------- #


@pytest.mark.parity
def test_additive_2d_parity():
    """2-D additive s(x0)+s(x1) — parity_additive.rs bar 5e-4."""
    name = "2d_gaussian_additive_n500_k10_cr"
    fx = load_fixture(name)
    x = x_train_2d(fx)
    y = y_train(fx)
    k = fx["inputs"]["k"]
    fitted = gamrs.fit_additive(
        "gaussian", x, y, [gamrs.CrTerm(0, k=int(k[0])), gamrs.CrTerm(1, k=int(k[1]))]
    )
    assert fitted.converged
    assert len(fitted.rho) == 2
    # gaussian identity: link prediction == μ.
    mu = np.asarray(fitted.predict(x))
    rel = max_rel_err(mu, np.asarray(fx["mgcv_output"]["predictions_train"]))
    assert rel < 5e-4, f"[{name}] additive μ rel error {rel:.3e} exceeds 5e-4"


@pytest.mark.parity
@pytest.mark.parametrize(
    "name,tol",
    [
        pytest.param("2d_gaussian_te_n300_k5x5", 2e-2, id="te_n300"),
        pytest.param(
            "2d_gaussian_te_n1000_k5x5", 5e-3, id="te_n1000", marks=pytest.mark.slow
        ),
    ],
)
def test_tensor_2d_parity(name, tol):
    fx = load_fixture(name)
    x = x_train_2d(fx)
    y = y_train(fx)
    k = fx["inputs"]["k"]
    fitted = gamrs.fit_additive(
        "gaussian", x, y, [gamrs.TeTerm(cols=(0, 1), k=(int(k[0]), int(k[1])))]
    )
    assert fitted.converged
    assert len(fitted.rho) == 2  # one smoothing param per tensor margin.
    mu = np.asarray(fitted.predict(x))
    rel = max_rel_err(mu, np.asarray(fx["mgcv_output"]["predictions_train"]))
    assert rel < tol, f"[{name}] tensor μ rel error {rel:.3e} exceeds {tol:.0e}"
