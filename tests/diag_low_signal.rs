//! Diagnostic: compare gamrs's fitted β coefficient-wise to mgcv's on the
//! worst-case parity fixture. Sign-flips in the centring basis cancel in
//! predictions but show up in β — so this test isolates *where* the gap is.
//!
//! Not part of the parity bar; uses `#[ignore]` so it runs only on demand
//! via `cargo test -p gamrs --test diag_low_signal -- --ignored --nocapture`.

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
    beta: Vec<f64>,
    lambda: Vec<f64>,
    predictions_train: Vec<f64>,
    scale: f64,
}

#[test]
#[ignore = "diagnostic; run with --ignored"]
fn diag_low_signal_beta_coefficientwise() {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/1d_gaussian_low_signal_n1000_k10_cr.json");
    let txt = std::fs::read_to_string(&p).unwrap();
    let fx: Fixture = serde_json::from_str(&txt).unwrap();
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];
    let fit = gamrs::fit(
        gamrs::family::gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .unwrap();
    let lambda_mgcv = fx.mgcv_output.lambda[0];
    let rho_mgcv = lambda_mgcv.ln();

    println!("\n=== low_signal_n1000 ===");
    let rho_scalar = fit.rho[0];
    println!("gamrs ρ̂   = {:.10}", rho_scalar);
    println!("mgcv ρ   = {:.10}", rho_mgcv);
    println!("Δρ       = {:.3e}", rho_scalar - rho_mgcv);
    println!();
    println!(
        "scale: gamrs={:.10e} mgcv={:.10e}",
        fit.scale, fx.mgcv_output.scale
    );
    println!();
    println!("β coefficient-wise:");
    println!("  idx | gamrs                | mgcv                | Δ");
    for i in 0..fit.beta.len() {
        let c = fit.beta[i];
        let m = fx.mgcv_output.beta[i];
        println!(
            "  {:>3} | {:>20.10e} | {:>20.10e} | {:>10.3e}",
            i,
            c,
            m,
            c - m
        );
    }
    println!();
    let pred = fit.predict(x.view()).unwrap();
    let max_pred_err = pred
        .iter()
        .zip(fx.mgcv_output.predictions_train.iter())
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1.0))
        .fold(0.0_f64, f64::max);
    println!("max pred rel err = {:.3e}", max_pred_err);
    println!();

    // If predictions differ but X·β_gamrs is identical to gamrs's prediction
    // (sanity), the gap is in the basis or β itself.
    // Sanity: predict reproduces fit.beta · design.
    let p0 = pred[0];
    let p_recomputed = fit
        .beta
        .iter()
        .enumerate()
        .map(|(i, b)| {
            if i == 0 {
                *b
            } else {
                0.0
            } // crude: just verify intercept lines up
        })
        .sum::<f64>();
    println!(
        "predict[0] = {:.6e}; β[0] (intercept) = {:.6e}",
        p0, p_recomputed
    );
}
