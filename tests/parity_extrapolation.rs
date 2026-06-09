//! Extrapolation PARITY vs mgcv.
//!
//! The CR parity fixtures carry mgcv's REML predictions at OUT-OF-RANGE x
//! (`inputs.x_extrap` / `mgcv_output.predictions_extrap`, on the μ / response
//! scale) — fields no test consumed until now. These assert gamrs's
//! extrapolation tracks mgcv's *actual out-of-range numbers* across identity
//! (gaussian), log (poisson) and logit (bernoulli) links.
//!
//! `tests/extrapolation_behavior.rs` checks the *shape* (linear-in-η, valid
//! range); this checks the *values* match the reference implementation.

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
    x_extrap: Vec<Vec<f64>>,
}
#[derive(Deserialize)]
struct MgcvOutput {
    predictions_extrap: Vec<f64>,
}

fn load_fixture(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn x_train_col(fx: &Fixture) -> (Array2<f64>, Array1<f64>, usize) {
    let xv: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = xv.len();
    (
        Array2::from_shape_vec((n, 1), xv).unwrap(),
        Array1::from(fx.inputs.y_train.clone()),
        fx.inputs.k[0],
    )
}

fn extrap_grid(fx: &Fixture) -> Array2<f64> {
    let xe: Vec<f64> = fx.inputs.x_extrap.iter().map(|r| r[0]).collect();
    let n = xe.len();
    Array2::from_shape_vec((n, 1), xe).unwrap()
}

/// Max element-wise relative error with a `|target|+1` denominator (matches
/// the in-range parity tests' metric; avoids divide-by-zero near 0).
fn max_rel(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target)
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1.0))
        .fold(0.0_f64, f64::max)
}

#[test]
fn gaussian_identity_extrap_parity() {
    let fx = load_fixture("1d_gaussian_smooth_n500_k10_cr");
    let (x, y, k) = x_train_col(&fx);
    let fit = gamrs::fit(
        gamrs::family::gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("gaussian fit");
    // identity link: predict() IS μ.
    let mu = fit.predict(extrap_grid(&fx).view()).unwrap();
    let rel = max_rel(mu.as_slice().unwrap(), &fx.mgcv_output.predictions_extrap);
    println!("[gaussian identity extrap] max_rel vs mgcv = {rel:.3e}");
    // Observed ~1e-8 (basis + REML λ match mgcv); 1e-6 leaves platform headroom.
    assert!(rel < 1e-6, "gaussian extrap rel {rel:.3e} exceeds 1e-6");
}

#[test]
fn poisson_log_extrap_parity() {
    let fx = load_fixture("1d_poisson_log_n300_k10_cr");
    let (x, y, k) = x_train_col(&fx);
    let fit =
        gamrs::fit(gamrs::family::poisson_log(), x.view(), y.view(), None, k).expect("poisson fit");
    // log link: μ = exp(η).
    let eta = fit.predict(extrap_grid(&fx).view()).unwrap();
    let mu: Vec<f64> = eta.iter().map(|&e| fit.link_kind.inverse(e)).collect();
    let rel = max_rel(&mu, &fx.mgcv_output.predictions_extrap);
    println!("[poisson log extrap] max_rel vs mgcv = {rel:.3e}");
    // Observed ~2e-4; 1e-3 leaves headroom for cross-platform REML/BLAS drift.
    assert!(rel < 1e-3, "poisson extrap rel {rel:.3e} exceeds 1e-3");
}

#[test]
fn bernoulli_logit_extrap_parity() {
    let fx = load_fixture("1d_bernoulli_logit_n300_k10_cr");
    let (x, y, k) = x_train_col(&fx);
    let fit = gamrs::fit(
        gamrs::family::bernoulli_logit(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("bernoulli fit");
    // logit link: μ = logit⁻¹(η).
    let eta = fit.predict(extrap_grid(&fx).view()).unwrap();
    let mu: Vec<f64> = eta.iter().map(|&e| fit.link_kind.inverse(e)).collect();
    let rel = max_rel(&mu, &fx.mgcv_output.predictions_extrap);
    println!("[bernoulli logit extrap] max_rel vs mgcv = {rel:.3e}");
    // Observed ~2e-3; 1e-2 leaves headroom for cross-platform REML/BLAS drift.
    assert!(rel < 1e-2, "bernoulli extrap rel {rel:.3e} exceeds 1e-2");
}
