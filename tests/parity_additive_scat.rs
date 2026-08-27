//! Parity test for gamrs's multi-smooth `Additive` path with the `scat`
//! (scaled-t) family against mgcv `scat()` 2-D / 3-D additive fixtures
//! (`y ~ s(x0) + s(x1) [+ s(x2)]`, identity link, heavy-tailed t noise).
//!
//! Closes the README's "scat / TDist multi-smooth … reference parity tests
//! are pending" gap. scat is the hardest convergence regime in the battery
//! (joint Newton over `[log λ…, log σ², log(ν−min.df)]`), so the bound is
//! relaxed relative to the GLM families — but it is a *measured*, locked
//! bound, not a smoke test. Both bounds tightened ~16× and ~8× when
//! `TDist::score_rank_adjustment`'s −1 came out (see `parity_scat.rs`).

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::tdist_identity;
use gamrs::fit_with_design;

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

fn max_rel_err(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1.0))
        .fold(0.0_f64, f64::max)
}

fn run_scat_additive(name: &str, d: usize, bound: f64) {
    let fx = load_fixture(name);
    let n = fx.inputs.x_train.len();
    let mut x_flat: Vec<f64> = Vec::with_capacity(n * d);
    for row in &fx.inputs.x_train {
        x_flat.extend_from_slice(&row[..d]);
    }
    let x = Array2::from_shape_vec((n, d), x_flat).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let y_mean = y.iter().sum::<f64>() / (n as f64);
    let y_var = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / (n as f64);

    let terms: Vec<TermSpec> = (0..d)
        .map(|j| TermSpec::Cr {
            col: j,
            k: fx.inputs.k[j],
        })
        .collect();
    // Seed ν = 5 (near scat's typical optimum), σ² = var(y)/10 — the joint
    // Newton profiles both. Mirrors the 1-D scat parity seeding.
    let fit = fit_with_design(
        tdist_identity(5.0, y_var * 0.1),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("Additive scat fit failed");
    assert_eq!(fit.rho.len(), d, "one smoothing param per term");

    // Identity link — gamrs predict returns μ directly (mgcv predictions are μ).
    let mu = fit.predict(x.view()).expect("predict failed");
    let rel = max_rel_err(mu.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    println!(
        "[additive scat {name}] max_rel = {rel:.3e}; σ̂² = {:.4}; edf = {:.2}; \
         iters = {}; converged = {}",
        fit.scale, fit.edf_total, fit.n_iters, fit.converged,
    );
    assert!(
        rel < bound,
        "additive scat μ rel error {rel:.3e} exceeds {bound:.1e}"
    );
}

#[test]
fn additive_2d_scat_n600_k8_cr() {
    // Bar 1.5e-3: observed 1.08e-3 (σ̂²=0.251 vs mgcv σ²=0.25, converged).
    //
    // This is the ONE fixture that moved the wrong way when the spurious
    // `∂ridge/∂ρ` term came out of the shape-aware ρ-gradient (5.7e-4 →
    // 1.08e-3), and it is not a regression in the fit: gamrs's ρ̂ went from
    // [3.809, 10.012] to [3.799, 10.077] against mgcv's log sp of
    // [3.736, 9.898], and its edf from 9.05 to 9.03 against mgcv's 8.9798 —
    // i.e. λ moved by ~7% on the second term while the total edf moved
    // TOWARD mgcv. The companion 3-D fixture improved 4× over the same
    // change (2.2e-3 → 5.7e-4) and `parity_scat_flat_ridge.rs` went from a
    // $291 gap to $22. The residual here is the ~1e-3 RELATIVE error that
    // still remains in scat's ρ-gradient at moderate ρ (measured in
    // `score_tests.rs::tdist_analytic_rho_grad_matches_fd`: analytic
    // +1.528723e-1 vs FD +1.530165e-1 at ρ=0) — a second-order term, not the
    // λ-envelope one.
    run_scat_additive("2d_scat_identity_n600_k8_cr", 2, 1.5e-3);
}

#[test]
fn additive_3d_scat_n800_k8_cr() {
    // Bar 1e-3: observed 5.7e-4 (σ̂²=0.162 vs mgcv σ²=0.162, converged).
    // Was 2.2e-3 against a 4e-3 bar until the spurious `∂ridge/∂ρ` term came
    // out of the shape-aware ρ-gradient; tightened 4× on that change.
    run_scat_additive("3d_scat_identity_n800_k8_cr", 3, 1e-3);
}
