//! Phase 2 parity test: gammon's `fit_scat_cr` (TDist + identity link +
//! joint outer Newton over `[log λ, log σ², log(ν-2)]`) vs mgcv on the
//! `1d_scat_unweighted_n300_k10_cr` fixture.
//!
//! Bound is intentionally relaxed (~5e-2 on μ) — the joint shape-param
//! outer is the hardest convergence regime in the gammon battery, and
//! mgcv's fixture doesn't expose its internal ν/σ² so we can't seed
//! gammon from the truth. We're asserting "the trait stack composes and
//! produces a sensible scat fit", not "byte-equivalent to mgcv".

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

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let s: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (s / n).sqrt()
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
fn scat_unweighted_n300_k10_cr() {
    let fx = load_fixture("1d_scat_unweighted_n300_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    // Seed Newton with sensible defaults: ν = 5 (close to scat's typical
    // optimum), σ² = var(y) / 10 (rough first-cut).
    let y_var = {
        let mean = y.iter().sum::<f64>() / (y.len() as f64);
        let var = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);
        var
    };
    let fit = gammon::fit(
        gammon::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("gammon::fit (scat/TDist) should not fail");
    assert!(fit.converged, "scat outer did not converge");

    let pred = fit.predict(x.view()).expect("predict should not fail");

    let rmse_v = rmse(pred.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    let rel = max_rel_err(pred.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    let abs_e = max_abs_err(pred.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    println!(
        "[scat unweighted n300] rmse = {rmse_v:.3e}; max_rel = {rel:.3e}; max_abs = {abs_e:.3e}; \
         ρ̂ = {:.3}; σ̂² = {:.4}; iters = {}; edf = {:.2}",
        fit.rho[0], fit.scale, fit.n_iters, fit.edf_total,
    );

    // Phase-2 bound: 5e-2 on μ relative — scat predictions are on the
    // identity scale, range is roughly ±1.5 (sin signal + noise). Mean
    // absolute prediction is ~0.3 so 5e-2 rel ≈ 0.015 absolute, which
    // is the wobble we'd expect from imperfect ν/σ² seeding.
    assert!(rel < 5e-2, "scat μ rel error {rel:.3e} exceeds 5e-2");
}
