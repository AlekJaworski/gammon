"""Fast smoke tier — every public Python API path exercised once on tiny data.

Includes explicit regression guards for the two bugs that shipped in 0.3.0 and
were patched in 0.3.1 (see the 2026-05-29 checkpoint):

1. ``gamrs.GAM(...).fit(x, y)`` raised ``TypeError`` on 1-D ``x`` because the
   native ``fit`` was lifted to ``PyReadonlyArray2`` but ``_low_level`` still
   passed 1-D. ``test_fit_accepts_1d_x`` locks the fix.
2. ``TpsTerm`` was not exported / not dispatchable. ``test_tpsterm_*`` lock it.

Both classes of bug would have been caught by *any* Python-level test — this
file is the standing guard the checkpoint called for.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------- #
# Package surface                                                             #
# --------------------------------------------------------------------------- #


def test_public_exports_present():
    for name in (
        "GAM",
        "Gam",
        "fit_additive",
        "CrTerm",
        "CrStableTerm",
        "ReTerm",
        "TeTerm",
        "TpsTerm",  # regression: must be importable (0.3.1).
    ):
        assert hasattr(gamrs, name), f"gamrs.{name} missing from public API"


# --------------------------------------------------------------------------- #
# Low-level GAM facade                                                        #
# --------------------------------------------------------------------------- #


def test_fit_accepts_1d_x(toy_gaussian):
    """Regression for the 0.3.0 ``TypeError: 'ndarray' cannot be cast as
    'ndarray'`` on 1-D ``x``. Must fit cleanly and expose k coefficients."""
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)  # x is 1-D — the bug case.
    assert g.converged_
    assert g.coef_.shape == (10,)


def test_fit_accepts_2d_column_x(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x.reshape(-1, 1), y)
    assert g.coef_.shape == (10,)


def test_predict_link_and_response(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    link = g.predict(x, scale="link")
    resp = g.predict(x, scale="response")
    assert link.shape == y.shape == resp.shape
    # Gaussian identity link: link == response.
    assert np.allclose(link, resp)


def test_predict_response_inverse_link_poisson(rng):
    x = np.linspace(0, 1, 300)
    mu = np.exp(0.5 + np.sin(2 * np.pi * x))
    y = rng.poisson(mu).astype(float)
    g = gamrs.GAM("poisson", k=10).fit(x, y)
    link = g.predict(x, scale="link")
    resp = g.predict(x, scale="response")
    # log link: response == exp(link), and all positive.
    assert np.allclose(resp, np.exp(link))
    assert np.all(resp > 0)


def test_predict_ci_ordering(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    mean, lo, hi = g.predict_ci(x, level=0.95)
    assert mean.shape == lo.shape == hi.shape == y.shape
    assert np.all(lo <= mean + 1e-9)
    assert np.all(mean <= hi + 1e-9)


def test_predict_diff_shapes(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    a = np.array([0.25, 0.5])
    b = np.array([0.75, 0.5])
    diff, lo, hi = g.predict_diff(a, b)
    assert diff.shape == (2,)
    assert np.all(lo <= diff + 1e-9) and np.all(diff <= hi + 1e-9)


def test_vcov_symmetric_psd(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    v = g.vcov()
    assert v.shape == (g.coef_.size, g.coef_.size)
    assert np.allclose(v, v.T, atol=1e-8)
    # PSD: smallest eigenvalue not meaningfully negative.
    assert np.linalg.eigvalsh(v).min() > -1e-8


def test_getters(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    assert isinstance(g.scale_, float) and g.scale_ > 0
    assert isinstance(g.edf_total_, float) and g.edf_total_ > 1.0
    assert isinstance(g.rho_, float)  # single smooth -> scalar.
    assert g.lambda_.shape == (1,)
    assert isinstance(g.n_iters_, int) and g.n_iters_ >= 1
    assert isinstance(g.converged_, bool)


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        gamrs.GAM("gaussian").predict(np.zeros(3))


def test_length_mismatch_raises(toy_gaussian):
    x, y = toy_gaussian
    with pytest.raises(ValueError):
        gamrs.GAM("gaussian").fit(x, y[:-1])


def test_bad_scale_raises(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    with pytest.raises(ValueError):
        g.predict(x, scale="nonsense")


# --------------------------------------------------------------------------- #
# Multi-smooth fit_additive + typed terms                                     #
# --------------------------------------------------------------------------- #


def test_fit_additive_two_cr_terms(rng):
    n = 300
    x = rng.uniform(0, 1, size=(n, 2))
    y = np.sin(2 * np.pi * x[:, 0]) + 0.5 * x[:, 1] ** 2 + rng.normal(0, 0.1, n)
    fitted = gamrs.fit_additive(
        "gaussian", x, y, [gamrs.CrTerm(0, k=8), gamrs.CrTerm(1, k=8)]
    )
    assert fitted.converged
    assert len(fitted.rho) == 2  # one smoothing param per term.
    pred = np.asarray(fitted.predict(x))
    assert pred.shape == (n,)


def test_fit_additive_requires_2d_x(rng):
    with pytest.raises(ValueError):
        gamrs.fit_additive("gaussian", np.linspace(0, 1, 10), np.zeros(10), [gamrs.CrTerm(0)])


def test_fit_additive_empty_terms_raises(rng):
    x = rng.uniform(0, 1, size=(10, 1))
    with pytest.raises(ValueError):
        gamrs.fit_additive("gaussian", x, np.zeros(10), [])


def test_tpsterm_default_k():
    # Regression: TpsTerm must construct and default k to 10 * len(cols).
    t = gamrs.TpsTerm(cols=(0, 1))
    assert t.k is None  # defaulted lazily at the FFI boundary.


def test_tpsterm_fit_dispatch(rng):
    """Regression: the ``"tp"`` basis branch must dispatch (0.3.1 fix)."""
    n = 200
    x = rng.uniform(0, 1, size=(n, 2))
    y = np.sin(3 * x[:, 0]) + np.cos(3 * x[:, 1]) + rng.normal(0, 0.1, n)
    fitted = gamrs.fit_additive("gaussian", x, y, [gamrs.TpsTerm(cols=(0, 1), k=15)])
    assert fitted.converged
    pred = np.asarray(fitted.predict(x))
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))


# --------------------------------------------------------------------------- #
# High-level Gam wrapper                                                       #
# --------------------------------------------------------------------------- #


def test_high_level_gam_smoke(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.Gam(predictors=["x0"], target="y", family="gaussian").fit(
        x.reshape(-1, 1), y
    )
    pred = np.asarray(g.predict(x.reshape(-1, 1)))
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))
    assert g.edf_total_ > 1.0


def test_term_string_cols_resolve_against_dataframe(rng):
    """CrTerm/TeTerm/TpsTerm/ReTerm accept string column names and resolve
    them against the DataFrame at fit time."""
    pd = pytest.importorskip("pandas")
    n = 200
    df = pd.DataFrame({
        "a": rng.uniform(0, 1, n),
        "b": rng.uniform(0, 1, n),
        "g": rng.integers(0, 5, n).astype(float),
    })
    df["y"] = np.sin(3 * df.a) + 0.5 * df.b**2 + rng.normal(0, 0.1, n)
    X = df[["a", "b", "g"]]

    # CrTerm with string column name
    g_str = gamrs.Gam(terms=[gamrs.CrTerm("a", k=10), gamrs.CrTerm("b", k=10)]).fit(
        X[["a", "b"]], df.y
    )
    # Equivalent fit with int indices
    g_int = gamrs.Gam(terms=[gamrs.CrTerm(0, k=10), gamrs.CrTerm(1, k=10)]).fit(
        X[["a", "b"]].values, df.y.values
    )
    np.testing.assert_allclose(
        np.asarray(g_str.predict(X[["a", "b"]])),
        np.asarray(g_int.predict(X[["a", "b"]].values)),
        rtol=1e-10,
        atol=1e-12,
    )

    # TeTerm + ReTerm with string names
    g_mix = gamrs.Gam(terms=[
        gamrs.TeTerm(cols=("a", "b"), k=(5, 5)),
        gamrs.ReTerm("g"),
    ]).fit(X, df.y)
    assert g_mix.converged_ is None or g_mix.converged_  # may be None for FS path
    assert g_mix.edf_total_ > 1.0


def test_term_string_col_unknown_raises(rng):
    """Unknown column names produce a clear error pointing at the term."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": rng.uniform(0, 1, 100)})
    df["y"] = df.a + rng.normal(0, 0.1, 100)
    with pytest.raises(ValueError, match="CrTerm.*'does_not_exist'"):
        gamrs.Gam(terms=[gamrs.CrTerm("does_not_exist")]).fit(df[["a"]], df.y)
