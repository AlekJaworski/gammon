"""Parity tier — high-dimensional (3-D … 10-D) additive Gaussian fits.

Mirrors ``tests/parity_highdim.rs`` at the Python layer: each fixture is an
all-CR additive Gaussian model (``y ~ s(x0) + … + s(x_{d-1})``, identity link,
REML). We fit through the public ``gamrs.fit_additive`` API and compare
response-scale predictions to ``mgcv_output.predictions_train``.

The bar (5e-4) matches the Rust ``parity_highdim.rs`` additive bound exactly:
the Python path delegates to the identical native fit, so any divergence beyond
the Rust bar is a real coercion/dispatch bug. Large-n cases (n >= 2000) carry
``pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

from conftest import load_fixture, max_rel_err, x_train_2d, y_train

pytestmark = pytest.mark.parity


# (fixture stem, mu_tol)
HIGHDIM_CASES = [
    pytest.param("3d_gaussian_mixed_n800_k10_cr", 5e-4, id="3d_n800"),
    pytest.param("4d_gaussian_mixed_n1000_k10_cr", 5e-4, id="4d_n1000"),
    pytest.param(
        "5d_gaussian_mixed_n1500_k8_cr", 5e-4, id="5d_n1500", marks=pytest.mark.slow
    ),
    pytest.param(
        "7d_neighbourhoods_compact_n3000", 5e-4, id="7d_n3000", marks=pytest.mark.slow
    ),
    pytest.param(
        "8d_neighbourhoods_like_n15000",
        5e-4,
        id="8d_n15000",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "10d_gaussian_n3000_k8_cr", 5e-4, id="10d_n3000", marks=pytest.mark.slow
    ),
]


@pytest.mark.parametrize("name,tol", HIGHDIM_CASES)
def test_highdim_additive_parity(name, tol):
    fx = load_fixture(name)
    x = x_train_2d(fx)
    y = y_train(fx)
    d = int(fx["inputs"]["d"])
    k = fx["inputs"]["k"]
    assert x.shape[1] == d

    terms = [gamrs.CrTerm(c, k=int(k[c])) for c in range(d)]
    fitted = gamrs.fit_additive("gaussian", x, y, terms)

    assert fitted.converged, f"[{name}] outer Newton did not converge"
    assert len(fitted.rho) == d, f"[{name}] expected one rho per term"

    # gaussian identity: link prediction == μ.
    mu = np.asarray(fitted.predict(x))
    rel = max_rel_err(mu, np.asarray(fx["mgcv_output"]["predictions_train"]))
    assert rel < tol, f"[{name}] additive μ rel error {rel:.3e} exceeds {tol:.0e}"
