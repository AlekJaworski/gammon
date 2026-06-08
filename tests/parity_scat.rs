//! Phase 2 parity test: gamrs's `fit_scat_cr` (TDist + identity link +
//! joint outer Newton over `[log λ, log σ², log(ν-2)]`) vs mgcv on the
//! `1d_scat_unweighted_n300_k10_cr` fixture.
//!
//! Bound is intentionally relaxed (~5e-2 on μ) — the joint shape-param
//! outer is the hardest convergence regime in the gamrs battery, and
//! mgcv's fixture doesn't expose its internal ν/σ² so we can't seed
//! gamrs from the truth. We're asserting "the trait stack composes and
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
    let fit = gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("gamrs::fit (scat/TDist) should not fail");
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

/// Raw-scale robustness via the **Rust API** (no Python-layer standardization).
/// scat with identity link is location-scale equivariant, so scaling the
/// response by a large constant `S` must scale the fitted μ by exactly `S`.
///
/// Before scat's standardization was relocated into the fit core, a raw-scale
/// response made the inner solve's `X'WX + λS` ill-conditioned — `W = ½·Dμμ ~
/// 1/σ² ~ 1/S²` shrinks `X'WX` far below `λS`, so the Cholesky degenerates and
/// the outer Newton stalls. This test pins that the Rust API now fits large-`y`
/// scat correctly (the conditioning fix that previously only the Python wrapper
/// applied). The unit-scale companion is `scat_unweighted_n300_k10_cr` above.
#[test]
fn scat_raw_scale_via_rust_api() {
    let fx = load_fixture("1d_scat_unweighted_n300_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let k = fx.inputs.k[0];

    // Scale the response up by 1e5 — comparable to a prices-style response
    // where var(y) ~ 1e10 and the raw-scale conditioning bug bites.
    const S: f64 = 1.0e5;
    let y_scaled = Array1::from_vec(fx.inputs.y_train.iter().map(|v| v * S).collect());
    let y_var = {
        let mean = y_scaled.iter().sum::<f64>() / (n as f64);
        y_scaled.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64)
    };
    // σ² seeded in *raw* (scaled) units, mirroring the unit-scale test's
    // var(y)/10 rough first-cut — the fit core standardizes internally.
    let fit = gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y_scaled.view(),
        None,
        k,
    )
    .expect("gamrs::fit (raw-scale scat) should not fail");
    assert!(fit.converged, "raw-scale scat outer did not converge");

    let pred = fit.predict(x.view()).expect("predict should not fail");
    // Equivariance: μ̂(S·y) = S·μ̂(y), so pred/S must match the unit-scale
    // mgcv reference within the same Phase-2 bound.
    let pred_unscaled: Vec<f64> = pred.iter().map(|p| p / S).collect();
    let rel = max_rel_err(&pred_unscaled, &fx.mgcv_output.predictions_train);
    println!(
        "[scat raw-scale S={S:.0e}] max_rel(pred/S vs mgcv) = {rel:.3e}; \
         σ̂² = {:.4e}; iters = {}",
        fit.scale, fit.n_iters,
    );
    assert!(
        rel < 5e-2,
        "raw-scale scat μ rel error {rel:.3e} exceeds 5e-2 (conditioning?)"
    );
}
