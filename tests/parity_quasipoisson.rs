//! Phase 4 parity test: gammon `fit_quasipoisson_cr` vs mgcv on 1-D
//! QuasiPoisson + log. Predictions on μ (count) scale; gammon's `predict`
//! returns η, we exponentiate.

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
    scale: f64,
}

fn load_fixture(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
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

#[test]
fn quasipoisson_log_n300_k10_cr() {
    let fx = load_fixture("1d_quasipoisson_log_n300_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    let fit = gammon::fit(
        gammon::family::quasipoisson_log(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("gammon::fit (QuasiPoisson) should not fail");
    assert!(fit.converged, "QuasiPoisson outer did not converge");

    let eta = fit.predict(x.view()).expect("predict failed");
    let mu_gammon: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let rel = max_rel_err(&mu_gammon, &fx.mgcv_output.predictions_train);
    let abs_e = max_abs_err(&mu_gammon, &fx.mgcv_output.predictions_train);
    let scale_rel = (fit.scale - fx.mgcv_output.scale).abs() / fx.mgcv_output.scale.max(1e-12);
    println!(
        "[quasipoisson n300 k10] max_rel = {rel:.3e}; max_abs = {abs_e:.3e}; \
         φ̂ gammon = {:.4} vs mgcv = {:.4} (rel {scale_rel:.3e}); \
         ρ̂ = {:.3}; iters = {}; edf = {:.2}",
        fit.scale, fx.mgcv_output.scale, fit.rho[0], fit.n_iters, fit.edf_total,
    );

    // Phase-4 bound: 5e-3 on predictions (same as Poisson), and 5e-2 on
    // φ̂ (looser because dispersion is profiled and the gammon-vs-mgcv
    // φ̂ alignment depends on the score's σ² convention).
    assert!(
        rel < 5e-3,
        "QuasiPoisson μ rel error {rel:.3e} exceeds 5e-3"
    );
    assert!(
        scale_rel < 5e-2,
        "QuasiPoisson φ̂ rel error {scale_rel:.3e} exceeds 5e-2"
    );
}
