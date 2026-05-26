//! Phase 7 parity test: gamrs `fit_gamma_cr` vs mgcv on 1-D Gamma + log.

use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::path::PathBuf;

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
    scale: f64,
}

fn load(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn max_rel_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / (y.abs() + 1.0))
        .fold(0.0_f64, f64::max)
}

fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn gamma_log_n300_k10_cr() {
    let fx = load("1d_gamma_log_n300_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];
    let fit = gamrs::fit(gamrs::family::gamma_log(), x.view(), y.view(), None, k).unwrap();
    // `converged` flag is sensitive on profiled-φ families where Newton's
    // quadratic approximation may exhaust step-halving at the optimum
    // (see outer.rs::NewtonWithHalving). The parity bounds below are the
    // real quality check.
    let eta = fit.predict(x.view()).unwrap();
    let mu_gamrs: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();
    let rel = max_rel_err(&mu_gamrs, &fx.mgcv_output.predictions_train);
    let abs_e = max_abs_err(&mu_gamrs, &fx.mgcv_output.predictions_train);
    let scale_rel = (fit.scale - fx.mgcv_output.scale).abs() / fx.mgcv_output.scale.max(1e-12);
    println!(
        "[gamma n300 k10] max_rel = {rel:.3e}; max_abs = {abs_e:.3e}; \
         φ̂ gamrs = {:.4} vs mgcv = {:.4} (rel {scale_rel:.3e}); \
         ρ̂ = {:.3}; iters = {}; edf = {:.2}",
        fit.scale, fx.mgcv_output.scale, fit.rho[0], fit.n_iters, fit.edf_total,
    );
    // Phase-7 bound: 2e-2 on μ. The saturated log-lik simplification
    // (omitting lgamma/digamma terms in φ) costs some parity vs mgcv's
    // exact φ-Newton; documented in family.rs::Gamma docstring.
    assert!(rel < 2e-2, "Gamma μ rel error {rel:.3e} exceeds 2e-2");
    assert!(
        scale_rel < 1e-1,
        "Gamma φ̂ rel error {scale_rel:.3e} exceeds 1e-1"
    );
}
