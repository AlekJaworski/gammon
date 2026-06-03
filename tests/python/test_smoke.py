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

import warnings

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


def test_discrete_true_emits_warning(rng):
    """discrete=True is a no-op for API compat; warns to keep users from
    silently mismatching their mgcv expectations."""
    x = rng.uniform(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    with pytest.warns(UserWarning, match="discrete=True is accepted"):
        gamrs.Gam(family="gaussian", discrete=True).fit(x, y)


def test_nthreads_emits_warning(rng):
    """nthreads is accepted for source compat but routes through the BLAS
    env vars in gamrs; warn so users aren't surprised it's a no-op."""
    x = rng.uniform(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    with pytest.warns(UserWarning, match="nthreads=4"):
        gamrs.Gam(family="gaussian", nthreads=4).fit(x, y)


def test_constant_column_auto_dropped(rng):
    """A constant predictor column (n_unique=1) is silently dropped on the
    predictors= path, exposed on dropped_predictors_, and a UserWarning is
    emitted. Predicts from a DataFrame containing the dropped column still
    work (it gets subselected away). Matches mgcv_rust 0.23.0 + mgcv R."""
    pd = pytest.importorskip("pandas")
    n = 200
    df = pd.DataFrame({
        "x0": rng.uniform(0, 10, n),
        "stories": np.full(n, 1.0),
        "x2": rng.uniform(-5, 5, n),
    })
    df["y"] = np.sin(df.x0) + 0.3 * df.x2 + rng.normal(0, 0.3, n)

    with pytest.warns(UserWarning, match="'stories' is constant"):
        g = gamrs.Gam(family="gaussian").fit(df[["x0", "stories", "x2"]], df.y)

    assert g.dropped_predictors_ == {"stories": 1.0}
    assert g._effective_predictors == ["x0", "x2"]
    # Predict from the original (3-col) DataFrame — the constant col gets subselected
    mu = g.predict(df[["x0", "stories", "x2"]])
    assert mu.shape == (n,)


def test_typed_terms_do_not_auto_drop(rng):
    """When the user is explicit via `terms=`, do NOT silently drop a
    constant column referenced by one of their terms — that would be more
    confusing than the natural error. (mgcv_rust auto-drop only fires on
    the predictors= mapping path.)"""
    pd = pytest.importorskip("pandas")
    n = 200
    df = pd.DataFrame({
        "x0": rng.uniform(0, 10, n),
        "stories": np.full(n, 1.0),
    })
    df["y"] = np.sin(df.x0) + rng.normal(0, 0.3, n)
    with pytest.raises(Exception):  # native raises a basis-construction error
        gamrs.Gam(
            terms=[gamrs.CrTerm("x0", k=10), gamrs.CrTerm("stories", k=10)]
        ).fit(df[["x0", "stories"]], df.y)


def test_all_constant_predictors_raises(rng):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": np.ones(100), "b": np.full(100, 2.0)})
    df["y"] = np.zeros(100)
    with pytest.raises(ValueError, match="all predictor columns are constant"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gamrs.Gam(family="gaussian").fit(df[["a", "b"]], df.y)


def test_default_gam_emits_no_warning(rng):
    """Sanity: a default Gam() does NOT warn — the warnings are only for
    the API-compat knobs that don't take effect."""
    x = rng.uniform(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        gamrs.Gam(family="gaussian").fit(x, y)


def test_parametric_term_recovers_slope(rng):
    """ParametricTerm fits the raw coefficient (no smoothing penalty).
    On `y = sin(x) + 2 * is_promo`, the parametric coefficient should
    land within a couple percent of the truth."""
    pd = pytest.importorskip("pandas")
    n = 1500
    x_smooth = rng.uniform(0, 10, n)
    is_promo = rng.integers(0, 2, n).astype(float)
    y = np.sin(x_smooth) + 2.0 * is_promo + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x": x_smooth, "is_promo": is_promo})
    X = df[["x", "is_promo"]]

    g = gamrs.Gam(terms=[
        gamrs.CrTerm("x", k=10),
        gamrs.ParametricTerm("is_promo"),
    ]).fit(X, y)

    # Parametric coef is the LAST one (terms concatenated in order).
    # Only the smooth has a smoothing parameter — lambda len == 1.
    assert len(g.lambda_) == 1
    assert abs(g.coef_[-1] - 2.0) < 0.05  # within 2.5% of truth
    assert g.converged_


def test_parametric_via_predictor_basis_map_matches_typed_term(rng):
    """Two ways to spell a parametric column: typed `ParametricTerm("x")`
    and `predictor_basis_map={"x": "parametric"}` produce identical fits."""
    pd = pytest.importorskip("pandas")
    n = 500
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "is_promo": rng.integers(0, 2, n).astype(float),
    })
    df["y"] = np.sin(df.x) + 1.5 * df.is_promo + rng.normal(0, 0.3, n)
    X = df[["x", "is_promo"]]

    g_typed = gamrs.Gam(terms=[
        gamrs.CrTerm("x", k=10),
        gamrs.ParametricTerm("is_promo"),
    ]).fit(X, df.y)
    g_map = gamrs.Gam(
        family="gaussian",
        predictor_basis_map={"is_promo": "parametric"},
    ).fit(X, df.y)
    np.testing.assert_allclose(g_typed.coef_, g_map.coef_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(g_typed.lambda_, g_map.lambda_, rtol=1e-10)


def test_n_unique_2_auto_promotes_to_parametric(rng):
    """A 0/1 indicator column on the implicit predictors= path is
    auto-promoted to ParametricTerm with a UserWarning (matches
    mgcv_rust + mgcv R smooth.r:1460 'reduce k' semantics)."""
    pd = pytest.importorskip("pandas")
    n = 600
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "is_promo": rng.integers(0, 2, n).astype(float),
    })
    df["y"] = np.sin(df.x) + 1.7 * df.is_promo + rng.normal(0, 0.3, n)
    X = df[["x", "is_promo"]]
    with pytest.warns(UserWarning, match="'is_promo'.*auto-promoted"):
        g = gamrs.Gam(family="gaussian").fit(X, df.y)
    # Only the smooth has a smoothing parameter
    assert len(g.lambda_) == 1
    assert abs(g.coef_[-1] - 1.7) < 0.05


def test_auto_k_works_with_parametric_terms(rng):
    """auto_k=True grows k on smooth terms but skips ParametricTerm (k=0
    placeholder; never grown). The parametric coefficient still recovers."""
    pd = pytest.importorskip("pandas")
    n = 1200
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "is_promo": rng.integers(0, 2, n).astype(float),
    })
    df["y"] = (
        np.sin(df.x * 1.5) + np.cos(df.x * 2.5)
        + 1.7 * df.is_promo + rng.normal(0, 0.2, n)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        g = gamrs.Gam(
            family="gaussian", auto_k=True, k_default=4, max_k_auto=20
        ).fit(df[["x", "is_promo"]], df.y)
    # x's k grew from 4 → something larger; is_promo stayed parametric (k=0)
    assert g._k_used[0] > 4
    assert g._k_used[1] == 0
    assert abs(g.coef_[-1] - 1.7) < 0.05


def test_parametric_linear_alias(rng):
    """`predictor_basis_map={"x": "linear"}` is the mgcv-user-friendly
    alias for `"parametric"` — same fit either way."""
    pd = pytest.importorskip("pandas")
    n = 300
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "z": rng.integers(0, 2, n).astype(float),
    })
    df["y"] = np.sin(df.x) + 0.7 * df.z + rng.normal(0, 0.3, n)
    X = df[["x", "z"]]
    g_p = gamrs.Gam(predictor_basis_map={"z": "parametric"}).fit(X, df.y)
    g_l = gamrs.Gam(predictor_basis_map={"z": "linear"}).fit(X, df.y)
    np.testing.assert_allclose(g_p.coef_, g_l.coef_, rtol=1e-12)


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


def test_json_roundtrip_preserves_predictions(rng):
    """Gam.to_json / Gam.from_json round-trip yields machine-epsilon
    identical predictions to the original fit, just like serialize/
    deserialize but in a human-debuggable plain-text form."""
    pd = pytest.importorskip("pandas")
    import json as stdlib_json
    n = 200
    df = pd.DataFrame({
        "x0": rng.uniform(0, 10, n),
        "x1": rng.uniform(-5, 5, n),
    })
    df["y"] = np.sin(df.x0) + 0.3 * df.x1**2 + rng.normal(0, 0.3, n)
    X = df[["x0", "x1"]]
    g = gamrs.Gam(
        terms=[gamrs.CrTerm("x0", k=10), gamrs.CrTerm("x1", k=12)]
    ).fit(X, df.y)

    payload = g.to_json()
    assert isinstance(payload, str)
    # Valid JSON
    parsed = stdlib_json.loads(payload)
    assert "beta" in parsed
    assert "edf_total" in parsed

    g_roundtrip = gamrs.Gam.from_json(payload)
    mu_orig = g.predict(X)
    mu_back = g_roundtrip.predict(X)
    # serde_json round-trips f64 through decimal — accept machine epsilon.
    np.testing.assert_allclose(mu_orig, mu_back, rtol=1e-12, atol=1e-14)


def test_subset_view_predict_ci(rng):
    """``gam[["x0"]].predict_ci(..., scale="deviation")`` returns the
    Wald CI on the masked η contribution. Same numbers as partial_effect."""
    pd = pytest.importorskip("pandas")
    n = 300
    df = pd.DataFrame({
        "x0": rng.uniform(0, 10, n),
        "x1": rng.uniform(-5, 5, n),
    })
    df["y"] = np.sin(df.x0) + 0.3 * df.x1**2 + rng.normal(0, 0.3, n)
    X = df[["x0", "x1"]]
    g = gamrs.Gam(
        terms=[gamrs.CrTerm("x0", k=10), gamrs.CrTerm("x1", k=15)]
    ).fit(X, df.y)

    # scale='deviation' is supported on subset views
    mean, lo, hi = g[["x0"]].predict_ci(X, level=0.95, scale="deviation")
    assert mean.shape == lo.shape == hi.shape == (n,)
    assert np.all(lo <= mean) and np.all(mean <= hi)

    # scale='link' includes the intercept when requested explicitly
    mean_int, _, _ = g[["x0", gamrs.Gam.INTERCEPT]].predict_ci(
        X, level=0.95, scale="link"
    )
    # difference between link-with-intercept and deviation should be a
    # constant (the intercept)
    diffs = mean_int - mean
    assert np.std(diffs) < 1e-8

    # scale='response' on a subset view is rejected with a clear message
    with pytest.raises(ValueError, match="only defined for full-model"):
        g[["x0"]].predict_ci(X, level=0.95, scale="response")

    # scale='deviation' without a subset view is also rejected
    with pytest.raises(ValueError, match="only meaningful on subset views"):
        g.predict_ci(X, level=0.95, scale="deviation")
