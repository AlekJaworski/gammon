"""Smoke tier — n-margin tensor product ``TeMultiTerm`` (mgcv ``te``) and
tensor interaction ``TiTerm`` (mgcv ``ti``) Python dispatch.

Mirrors ``tests/tensor_multi_smoke.rs`` at the Python layer: the typed
terms must import, dispatch through ``gamrs.fit_additive`` to the native
n-margin builders, converge, and produce finite predictions. No mgcv
n-margin te/ti fixture exists, so this is a smoke + equivalence guard, not
a numerical-parity gate.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

pytestmark = pytest.mark.smoke


def test_te_multi_and_ti_exported():
    # Regression: typed terms must be importable / re-exported.
    for name in ("TeMultiTerm", "TiTerm"):
        assert hasattr(gamrs, name), f"gamrs.{name} not exported"


def test_te_multi_3margin_fit(rng):
    n = 400
    x = rng.uniform(0, 1, size=(n, 3))
    y = (
        np.sin(2 * np.pi * x[:, 0])
        + (1.5 * x[:, 1] - 0.5) ** 2
        + 0.8 * x[:, 0] * x[:, 2]
        + rng.normal(0, 0.15, n)
    )
    fitted = gamrs.fit_additive(
        "gaussian", x, y, [gamrs.TeMultiTerm(cols=(0, 1, 2), k=(4, 4, 4))]
    )
    assert fitted.converged
    assert len(fitted.rho) == 3  # one smoothing param per margin (D=3).
    pred = np.asarray(fitted.predict(x))
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))


def test_te_multi_default_k():
    # k defaults lazily at the FFI boundary (None on the dataclass).
    t = gamrs.TeMultiTerm(cols=(0, 1))
    assert t.k is None


def test_ti_3margin_fit(rng):
    n = 400
    x = rng.uniform(0, 1, size=(n, 3))
    y = (
        np.sin(2 * np.pi * x[:, 0])
        + (1.5 * x[:, 1] - 0.5) ** 2
        + 0.5 * x[:, 1] * x[:, 2]
        + rng.normal(0, 0.15, n)
    )
    fitted = gamrs.fit_additive(
        "gaussian", x, y, [gamrs.TiTerm(cols=(0, 1, 2), k=(4, 4, 4))]
    )
    assert fitted.converged
    assert len(fitted.rho) == 3
    pred = np.asarray(fitted.predict(x))
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))


def test_te_multi_2margin_matches_te(rng):
    """A 2-margin TeMultiTerm must match the 2-margin TeTerm path to FP —
    proving the generalization reduces to the validated 2-margin builder."""
    n = 350
    x = rng.uniform(0, 1, size=(n, 2))
    y = (
        np.sin(2 * np.pi * x[:, 0])
        + (1.5 * x[:, 1] - 0.5) ** 2
        + 0.7 * x[:, 0] * x[:, 1]
        + rng.normal(0, 0.12, n)
    )
    fit_te = gamrs.fit_additive("gaussian", x, y, [gamrs.TeTerm(cols=(0, 1), k=(5, 5))])
    fit_multi = gamrs.fit_additive(
        "gaussian", x, y, [gamrs.TeMultiTerm(cols=(0, 1), k=(5, 5))]
    )
    p_te = np.asarray(fit_te.predict(x))
    p_multi = np.asarray(fit_multi.predict(x))
    max_diff = float(np.max(np.abs(p_te - p_multi)))
    assert max_diff < 1e-9, f"TeMulti(2col) vs TeTerm max diff {max_diff:.3e}"
