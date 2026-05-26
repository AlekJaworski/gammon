//! Phase 1 parity test: gammon's `fit_binomial_cr` vs mgcv on 1-D Bernoulli +
//! logit fixtures. Mirrors `parity_gaussian.rs` but expects μ-scale
//! predictions (gammon's `predict` returns η for the Bernoulli path; we
//! apply the inverse link to compare against mgcv's `predictions_train`,
//! which is on the μ / probability scale).

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    mgcv_output: MgcvOutput,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
}

#[derive(Deserialize)]
struct MgcvOutput {
    predictions_train: Vec<f64>,
}

fn load_fixture(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn max_rel_err(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1.0))
        .fold(0.0_f64, f64::max)
}

fn max_abs_err(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let s: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (s / n).sqrt()
}

fn fit_and_check(fixture_name: &str, max_rel_pred: f64) {
    let fx = load_fixture(fixture_name);
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    let fit = gammon::fit(gammon::family::bernoulli_logit(), x.view(), y.view(), None, k)
        .unwrap_or_else(|e| panic!("[{fixture_name}] fit failed: {e}"));
    assert!(fit.converged, "[{fixture_name}] outer did not converge");

    // gammon's `predict` for Bernoulli returns η = X·β (link scale). mgcv's
    // `predictions_train` is on the μ scale — apply inverse logit.
    let eta = fit
        .predict(x.view())
        .unwrap_or_else(|e| panic!("predict failed: {e}"));
    let mu_gammon: Vec<f64> = eta.iter().map(|&e| logistic(e)).collect();

    let rel = max_rel_err(&mu_gammon, &fx.mgcv_output.predictions_train);
    let abs_e = max_abs_err(&mu_gammon, &fx.mgcv_output.predictions_train);
    let rmse_v = rmse(&mu_gammon, &fx.mgcv_output.predictions_train);
    println!(
        "[{fixture_name}] max_rel = {rel:.3e}; max_abs = {abs_e:.3e}; rmse = {rmse_v:.3e}; ρ̂ = {:.3}; iters = {}; edf = {:.2}",
        fit.rho[0], fit.n_iters, fit.edf_total,
    );
    assert!(
        rel < max_rel_pred,
        "[{fixture_name}] μ max-rel error {rel:.3e} exceeds {max_rel_pred:.3e}",
    );
}

// Phase-1 bound: 2e-3 on μ-scale predictions. Observed worst case ~1e-3
// (1d_bernoulli_logit_n300_k10_cr). Tighter than Gaussian's because
// Bernoulli has fixed σ² = 1 — no profile/gradient inconsistency. The
// remaining gap is PIRLS-Newton-outer micro-precision: gammon's ρ̂ lands
// ~1e-2 short of mgcv's on these fixtures, scaled up by PIRLS μ
// sensitivity ⇒ ~1e-3 in μ. Closing further needs analytic Hessian on
// the outer or mgcv-exact PIRLS halving rules; deferred.
const REL_PRED: f64 = 2e-3;

#[test]
fn bernoulli_logit_n300_k10_cr() {
    fit_and_check("1d_bernoulli_logit_n300_k10_cr", REL_PRED);
}

#[test]
fn bernoulli_logit_n1000_k10_cr() {
    fit_and_check("1d_bernoulli_logit_n1000_k10_cr", REL_PRED);
}
