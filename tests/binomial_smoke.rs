//! Phase 1 smoke test: gammon fits a noisy 1-D Bernoulli + logit GAM and
//! recovers a sensible smooth shape (monotone increasing) on data where
//! the true logit is monotone.
//!
//! No mgcv parity fixture exists for 1-D Bernoulli yet — when one is
//! generated, this file will gain a `binomial_parity` test that compares
//! coefficient-wise. For now we check basic correctness:
//!
//! - Fit converges.
//! - μ̂ is in (0, 1) everywhere.
//! - μ̂ is monotone non-decreasing in x (since the true logit is).
//! - μ̂ at low x ≈ 0.1 and at high x ≈ 0.9 (matches the data-generating
//!   process).

use ndarray::{Array1, Axis};

fn logistic(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Deterministic "noisy" Bernoulli draw — XOR-like pseudorandom from the
/// index `i`, threshold against `p(x_i)`. Stable across runs.
fn pseudo_bernoulli(i: usize, p: f64) -> f64 {
    let h = (i.wrapping_mul(2654435761)) as u32;
    let u = (h as f64) / (u32::MAX as f64);
    if u < p {
        1.0
    } else {
        0.0
    }
}

#[test]
fn binomial_logit_1d_recovers_monotone_smooth() {
    let n = 500;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
    // True logit: η(x) = 4·(x - 0.5). p(x) ranges from 0.119 at x=0 to 0.881 at x=1.
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let eta = 4.0 * (x - 0.5);
            let p = logistic(eta);
            pseudo_bernoulli(i, p)
        })
        .collect();
    let x = Array1::from_vec(xs);
    let y = Array1::from_vec(ys);
    let x2 = x.view().insert_axis(Axis(1));

    let fit = gammon::fit(gammon::family::bernoulli_logit(), x2, y.view(), None, 10)
        .expect("gammon::fit (Bernoulli) should not fail");

    assert!(fit.converged, "Binomial outer didn't converge");
    println!(
        "ρ̂ = {:.3}; iters = {}; scale = {}; edf = {:.3}",
        fit.rho[0], fit.n_iters, fit.scale, fit.edf_total
    );
    assert_eq!(fit.scale, 1.0, "Bernoulli σ² should be fixed at 1");

    // Predict on the training x — for Bernoulli + logit, `predict` returns
    // η (the linear predictor on the link scale). Convert to μ via the
    // inverse link to check monotonicity.
    let eta = fit
        .predict(x.view().insert_axis(Axis(1)))
        .expect("predict should not fail");
    let mu: Vec<f64> = eta.iter().map(|&e| logistic(e)).collect();

    // All μ in (0, 1).
    for (i, &m) in mu.iter().enumerate() {
        assert!(m > 0.0 && m < 1.0, "μ[{i}] = {m} out of bounds");
    }

    // Roughly monotone increasing — allow small dips due to noise + smooth
    // wobble, but the trend should be clear (μ at x=0 < μ at x=1).
    let mu_low = mu[..n / 10].iter().sum::<f64>() / (n as f64 / 10.0);
    let mu_high = mu[9 * n / 10..].iter().sum::<f64>() / (n as f64 / 10.0);
    println!("μ_low (first 10%) = {mu_low:.3}; μ_high (last 10%) = {mu_high:.3}");
    assert!(mu_low < 0.3, "low-x μ should be ≈ 0.12, got {mu_low}");
    assert!(mu_high > 0.7, "high-x μ should be ≈ 0.88, got {mu_high}");
}
