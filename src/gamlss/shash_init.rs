//! `shash` GAMLSS initialisation (TDD phase 3).
//!
//! Ports mgcv's `shash` `initialize` expression (`gamlss.r:3974-4024`): start
//! the four linear predictors from cheap penalised regressions before the
//! joint Newton iteration (phase 4) refines them.
//!
//!   1. β₁ (location μ, identity link) = ridge regression of `y` on `Xμ`.
//!   2. β₂ (log-scale τ, `logeb` link)  = ridge regression of `log|y − μ̂|`
//!      on `Xτ`, with `μ̂ = Xμ·β₁` (the init treats the `logeb` offset `b` as
//!      negligible, regressing the linear predictor directly on `log|resid|`).
//!   3. β₃ (skewness ε) and β₄ (log-kurtosis φ): identity links regressed on
//!      `linkfun(0) = 0`, i.e. the zero vector — so both start at **0**.
//!
//! mgcv performs each regression with `pen.reg`, whose square-root penalty `E`
//! is used purely as a *regulariser* for the warm start (mgcv's own comment:
//! "best we can do here is to use E only as a regularizer"), with an adaptive
//! penalty-weight loop that targets an effective-dof near `rank(X)`. That loop
//! is a heuristic for the starting point only — the joint Newton refines it and
//! phase-6 parity does not depend on its exact value — so we do **not**
//! replicate it. Instead [`pen_ols`] is a ridge-stabilised least squares that
//! reduces **exactly** to `pen.reg`'s zero-penalty (`qr.coef`/OLS) limit, the
//! well-defined case we confront mgcv on bit-for-bit (see the tests). A nonzero
//! `ridge` only shrinks toward zero for conditioning, never changing that limit.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::error::Result;
use crate::inner::{CholeskySolver, LinearSolver};

/// Starting coefficient blocks for the four shash linear predictors
/// `(μ, τ, ε, φ)`, in coefficient (β) space — directly the `start` vector
/// mgcv assembles, split per linear predictor.
#[derive(Clone, Debug)]
pub struct ShashInit {
    /// Location β₁ (identity link).
    pub beta_mu: Array1<f64>,
    /// Log-scale β₂ (`logeb` link).
    pub beta_tau: Array1<f64>,
    /// Skewness β₃ (identity link) — all zero.
    pub beta_eps: Array1<f64>,
    /// Log-kurtosis β₄ (identity link) — all zero.
    pub beta_phi: Array1<f64>,
}

/// Ridge-stabilised least squares: solve `(XᵀX + ridge·I) β = Xᵀz` via
/// Cholesky. With `ridge == 0` and a full-column-rank `X` this is ordinary
/// least squares, reproducing mgcv `pen.reg`'s `qr.coef` (zero-penalty) limit.
fn pen_ols(x: ArrayView2<f64>, z: ArrayView1<f64>, ridge: f64) -> Result<Array1<f64>> {
    let p = x.ncols();
    let mut a = x.t().dot(&x); // XᵀX  (p×p, SPD when X is full column rank)
    if ridge != 0.0 {
        for i in 0..p {
            a[[i, i]] += ridge;
        }
    }
    let xtz = x.t().dot(&z); // Xᵀz  (p)
    let fact = CholeskySolver::factorize(a)?;
    Ok(CholeskySolver::solve(&fact, xtz.view()))
}

/// `log|y − μ̂|`, the τ-block regression target. Guards the (measure-zero)
/// exact-zero residual — where mgcv would produce `-Inf` then `NaN→0` via
/// `pen.reg`'s finite-coefficient cleanup — with the smallest representable
/// log instead; on continuous data this never triggers, so the OLS limit
/// still matches mgcv bit-for-bit.
fn log_abs_resid(y: ArrayView1<f64>, mu_hat: ArrayView1<f64>) -> Array1<f64> {
    y.iter()
        .zip(mu_hat.iter())
        .map(|(&yi, &mi)| {
            let r = (yi - mi).abs();
            if r > 0.0 {
                r.ln()
            } else {
                f64::MIN_POSITIVE.ln()
            }
        })
        .collect()
}

/// Initialise the four shash coefficient blocks from the per-predictor design
/// matrices and the response, per mgcv's `shash` `initialize`. `ridge ≥ 0` is a
/// conditioning stabiliser (use `0.0` for the exact OLS/`pen.reg` zero-penalty
/// limit).
pub fn shash_init(
    x_mu: ArrayView2<f64>,
    x_tau: ArrayView2<f64>,
    x_eps: ArrayView2<f64>,
    x_phi: ArrayView2<f64>,
    y: ArrayView1<f64>,
    ridge: f64,
) -> Result<ShashInit> {
    // 1) location: ridge regression of y on Xμ (identity link).
    let beta_mu = pen_ols(x_mu, y, ridge)?;
    let mu_hat = x_mu.dot(&beta_mu);

    // 2) log-scale: ridge regression of log|resid| on Xτ.
    let lres = log_abs_resid(y, mu_hat.view());
    let beta_tau = pen_ols(x_tau, lres.view(), ridge)?;

    // 3) skewness + log-kurtosis: identity links on linkfun(0)=0 ⇒ zeros.
    let beta_eps = Array1::zeros(x_eps.ncols());
    let beta_phi = Array1::zeros(x_phi.ncols());

    Ok(ShashInit {
        beta_mu,
        beta_tau,
        beta_eps,
        beta_phi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use std::path::PathBuf;

    /// mgcv's `shash` `initialize` evaluated in the zero-penalty (OLS) limit —
    /// see `scripts/r/gen_shash_init_fixture.R`. Row-major design blocks +
    /// response + the resulting `start` blocks.
    #[derive(serde::Deserialize)]
    struct InitFixture {
        n: usize,
        p: Blocks,
        #[serde(rename = "X_mu")]
        x_mu: Vec<f64>,
        #[serde(rename = "X_tau")]
        x_tau: Vec<f64>,
        #[serde(rename = "X_eps")]
        x_eps: Vec<f64>,
        #[serde(rename = "X_phi")]
        x_phi: Vec<f64>,
        y: Vec<f64>,
        lres: Vec<f64>,
        beta_mu: Vec<f64>,
        beta_tau: Vec<f64>,
        beta_eps: Vec<f64>,
        beta_phi: Vec<f64>,
    }

    #[derive(serde::Deserialize)]
    struct Blocks {
        mu: usize,
        tau: usize,
        eps: usize,
        phi: usize,
    }

    fn load_fixture() -> InitFixture {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/shash_init_mgcv.json");
        let txt = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
        serde_json::from_str(&txt).expect("malformed shash_init fixture json")
    }

    fn mat(flat: &[f64], rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), flat.to_vec()).expect("shape")
    }

    fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "length mismatch");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    // --- exact confrontation with mgcv in the zero-penalty (OLS) limit ------

    #[test]
    fn init_matches_mgcv_ols_limit() {
        let fx = load_fixture();
        let x_mu = mat(&fx.x_mu, fx.n, fx.p.mu);
        let x_tau = mat(&fx.x_tau, fx.n, fx.p.tau);
        let x_eps = mat(&fx.x_eps, fx.n, fx.p.eps);
        let x_phi = mat(&fx.x_phi, fx.n, fx.p.phi);
        let y = Array1::from(fx.y.clone());

        let init = shash_init(
            x_mu.view(),
            x_tau.view(),
            x_eps.view(),
            x_phi.view(),
            y.view(),
            0.0, // OLS limit — must match mgcv pen.reg(E=0)=qr.coef bit-for-bit
        )
        .expect("shash_init");

        // Intermediate: the log|resid| target must match mgcv's `lres` exactly
        // (validates the identity μ-link + residual composition before the
        // second regression even runs).
        let mu_hat = x_mu.dot(&init.beta_mu);
        let lres = log_abs_resid(y.view(), mu_hat.view());
        assert!(
            max_abs_err(lres.as_slice().unwrap(), &fx.lres) < 1e-9,
            "log|resid| vs mgcv: max err {}",
            max_abs_err(lres.as_slice().unwrap(), &fx.lres)
        );

        assert!(
            max_abs_err(init.beta_mu.as_slice().unwrap(), &fx.beta_mu) < 1e-7,
            "β_mu vs mgcv: {:?} vs {:?}",
            init.beta_mu,
            fx.beta_mu
        );
        assert!(
            max_abs_err(init.beta_tau.as_slice().unwrap(), &fx.beta_tau) < 1e-7,
            "β_tau vs mgcv: {:?} vs {:?}",
            init.beta_tau,
            fx.beta_tau
        );
        // ε, φ blocks start at exactly zero (identity link on linkfun(0)=0).
        assert!(
            max_abs_err(init.beta_eps.as_slice().unwrap(), &fx.beta_eps) < 1e-12,
            "β_eps must be zero"
        );
        assert!(
            max_abs_err(init.beta_phi.as_slice().unwrap(), &fx.beta_phi) < 1e-12,
            "β_phi must be zero"
        );
    }

    #[test]
    fn shape_blocks_are_zero_and_sized() {
        let fx = load_fixture();
        let x_mu = mat(&fx.x_mu, fx.n, fx.p.mu);
        let x_tau = mat(&fx.x_tau, fx.n, fx.p.tau);
        let x_eps = mat(&fx.x_eps, fx.n, fx.p.eps);
        let x_phi = mat(&fx.x_phi, fx.n, fx.p.phi);
        let y = Array1::from(fx.y.clone());
        let init = shash_init(
            x_mu.view(),
            x_tau.view(),
            x_eps.view(),
            x_phi.view(),
            y.view(),
            1e-3,
        )
        .expect("shash_init");
        assert_eq!(init.beta_eps.len(), fx.p.eps);
        assert_eq!(init.beta_phi.len(), fx.p.phi);
        assert!(init.beta_eps.iter().all(|&v| v == 0.0));
        assert!(init.beta_phi.iter().all(|&v| v == 0.0));
    }

    // --- property: a positive ridge shrinks the coefficients toward zero ----

    #[test]
    fn ridge_shrinks_coefficients() {
        let fx = load_fixture();
        let x_mu = mat(&fx.x_mu, fx.n, fx.p.mu);
        let x_tau = mat(&fx.x_tau, fx.n, fx.p.tau);
        let x_eps = mat(&fx.x_eps, fx.n, fx.p.eps);
        let x_phi = mat(&fx.x_phi, fx.n, fx.p.phi);
        let y = Array1::from(fx.y.clone());

        let l2 = |b: &Array1<f64>| b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let ols = shash_init(
            x_mu.view(),
            x_tau.view(),
            x_eps.view(),
            x_phi.view(),
            y.view(),
            0.0,
        )
        .unwrap();
        let ridged = shash_init(
            x_mu.view(),
            x_tau.view(),
            x_eps.view(),
            x_phi.view(),
            y.view(),
            10.0, // heavy ridge to make shrinkage unambiguous
        )
        .unwrap();
        assert!(
            l2(&ridged.beta_mu) < l2(&ols.beta_mu),
            "ridge should shrink β_mu: {} !< {}",
            l2(&ridged.beta_mu),
            l2(&ols.beta_mu)
        );
    }

    // --- property: recovers the truth on data drawn from a Gaussian (ε=φ=0) -

    fn frac(v: f64) -> f64 {
        v - v.floor()
    }
    fn pnormal(u1: f64, u2: f64) -> f64 {
        (-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// mgcv's shash init (like the two-stage quantile path) regresses
    /// `log|resid|` *without* the Euler bias correction, so the τ intercept
    /// estimates `log σ + E[log|N(0,1)|]`. We add that constant back to recover
    /// σ. (Matches `_E_LOG_ABS_NORMAL` in `python/gamrs/_quantile.py`.)
    const E_LOG_ABS_NORMAL: f64 = -0.6351814227307388;

    #[test]
    fn recovers_gaussian_location_and_scale() {
        // y = β·x + σ·N(0,1) with a known constant σ; ε=φ=0 ground truth.
        let n = 4000usize;
        let beta_true = [0.7_f64, 1.3, -0.5];
        let sigma_true = 0.45_f64;

        let mut x_mu = Array2::<f64>::zeros((n, 3));
        let mut y = Array1::<f64>::zeros(n);
        // τ/ε/φ designs are intercept-only here (constant scale/shape).
        let x_int = Array2::<f64>::ones((n, 1));
        for i in 0..n {
            let x1 = frac((i as f64 + 0.5) * 0.6180339887498949);
            let x2 = frac((i as f64 + 0.5) * 0.3819660112501051 + 0.137);
            x_mu[[i, 0]] = 1.0;
            x_mu[[i, 1]] = x1;
            x_mu[[i, 2]] = x2;
            let mu = beta_true[0] + beta_true[1] * x1 + beta_true[2] * x2;
            let u1 = frac((i as f64 + 0.5) * 0.7548776662466927 + 0.31);
            let u2 = frac((i as f64 + 0.5) * 0.5698402909980532 + 0.59);
            y[i] = mu + sigma_true * pnormal(u1, u2);
        }

        let init = shash_init(
            x_mu.view(),
            x_int.view(),
            x_int.view(),
            x_int.view(),
            y.view(),
            0.0,
        )
        .expect("shash_init");

        // Location coefficients recover the truth within sampling error.
        for (k, &bt) in beta_true.iter().enumerate() {
            assert!(
                (init.beta_mu[k] - bt).abs() < 0.05,
                "β_mu[{k}] = {} vs true {bt}",
                init.beta_mu[k]
            );
        }
        // τ intercept, bias-corrected, recovers log σ ⇒ exp recovers σ.
        let sigma_hat = (init.beta_tau[0] - E_LOG_ABS_NORMAL).exp();
        assert!(
            (sigma_hat - sigma_true).abs() / sigma_true < 0.1,
            "σ̂ (bias-corrected) = {sigma_hat} vs true {sigma_true}"
        );
    }
}
