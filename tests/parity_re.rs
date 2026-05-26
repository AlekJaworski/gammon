//! Smoke tests for the `Re` (random-effect) `DesignStrategy`.
//!
//! These are NOT mgcv-parity fixtures — they verify that
//! `gammon::fit_with_design(family, Re, x, y, None)` produces sensible
//! group-mean estimates on synthetic data with known groupings:
//!
//! 1. Per-group fitted μ̂ ≈ per-group y mean (with mild shrinkage from the
//!    REML-selected λ).
//! 2. Predicting on a known level returns the level's fitted μ̂.
//! 3. Predicting on an UNSEEN level returns the intercept only (the RE
//!    one-hot is zero for unseen levels — matches mgcv `bs="re"`).
//! 4. Random-effect derivative is zero (step function over a categorical
//!    predictor — `predict_deriv` returns identically zero).

use ndarray::{Array1, Axis};

/// Synthetic data: 4 groups, 50 obs each (n=200). Group means are
/// `[1.0, 3.0, 5.0, 7.0]`; per-obs noise σ = 0.1 so the group structure
/// dominates.
fn synth() -> (Array1<f64>, Array1<f64>, Vec<f64>) {
    use std::cell::Cell;

    // Deterministic LCG so the tests are reproducible without an RNG dep.
    let state = Cell::new(123_456_789_u64);
    let next_gauss = || {
        // Box-Muller from two LCG-uniform draws.
        let lcg = || {
            let s = state
                .get()
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            state.set(s);
            ((s >> 11) as f64) / (1u64 << 53) as f64
        };
        let u1 = lcg().max(1e-12);
        let u2 = lcg();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let means = [1.0, 3.0, 5.0, 7.0];
    let per_group = 50usize;
    let n = means.len() * per_group;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let group_ids: Vec<f64> = (0..means.len()).map(|i| (i + 1) as f64).collect();
    for (g, &mu) in means.iter().enumerate() {
        for _ in 0..per_group {
            x.push(group_ids[g]);
            y.push(mu + 0.1 * next_gauss());
        }
    }
    (Array1::from_vec(x), Array1::from_vec(y), group_ids)
}

fn group_mean(y: &Array1<f64>, x: &Array1<f64>, group: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..y.len() {
        if x[i] == group {
            sum += y[i];
            count += 1;
        }
    }
    sum / count as f64
}

#[test]
fn re_basis_recovers_group_means() {
    let (x, y, group_ids) = synth();
    let x2 = x.view().insert_axis(Axis(1));
    let fit = gammon::fit_with_design(
        gammon::family::gaussian_identity(),
        gammon::Re,
        x2,
        y.view(),
        None,
    )
    .expect("fit failed");
    assert!(fit.converged, "outer Newton did not converge");

    // Predict at each known group level — μ̂_g should be close to the
    // empirical y-mean for that group (with mild shrinkage from the
    // REML-selected λ — bound is loose to absorb shrinkage).
    let probe = Array1::from_vec(group_ids.clone());
    let mu_hat = fit
        .predict(probe.view().insert_axis(Axis(1)))
        .expect("predict failed");
    for (i, &g) in group_ids.iter().enumerate() {
        let emp = group_mean(&y, &x, g);
        let pred = mu_hat[i];
        let abs_err = (pred - emp).abs();
        // Empirical group mean ~ true_mean ± 0.1/sqrt(50) ≈ 0.014.
        // Shrinkage toward the grand mean is bounded by ~|emp - grand_mean|·λ/(1+λ);
        // at moderate λ the shrinkage stays under ~0.5 on the per-group mean.
        assert!(
            abs_err < 0.6,
            "group {g}: |μ̂={pred:.4} − ȳ_g={emp:.4}| = {abs_err:.3e} too large"
        );
    }
}

#[test]
fn re_unseen_level_returns_intercept() {
    let (x, y, _group_ids) = synth();
    let x2 = x.view().insert_axis(Axis(1));
    let fit = gammon::fit_with_design(
        gammon::family::gaussian_identity(),
        gammon::Re,
        x2,
        y.view(),
        None,
    )
    .expect("fit failed");

    // Level 99.0 was NOT in training. RE one-hot is all-zero for unseen
    // → prediction = intercept (β[0]).
    let unseen = Array1::from_vec(vec![99.0]);
    let pred = fit
        .predict(unseen.view().insert_axis(Axis(1)))
        .expect("predict failed");
    let intercept = fit.beta[0];
    let diff = (pred[0] - intercept).abs();
    assert!(
        diff < 1e-12,
        "unseen-level prediction {:.6e} should equal intercept {:.6e}; diff {:.3e}",
        pred[0],
        intercept,
        diff
    );
}

#[test]
fn re_predict_deriv_is_zero() {
    let (x, y, group_ids) = synth();
    let x2 = x.view().insert_axis(Axis(1));
    let fit = gammon::fit_with_design(
        gammon::family::gaussian_identity(),
        gammon::Re,
        x2,
        y.view(),
        None,
    )
    .expect("fit failed");

    // The RE basis is a step function over a categorical predictor; the
    // derivative w.r.t. the grouping variable is identically zero.
    let probe = Array1::from_vec(group_ids);
    let d = fit
        .predict_deriv(probe.view().insert_axis(Axis(1)))
        .expect("predict_deriv failed");
    for (i, &v) in d.iter().enumerate() {
        assert_eq!(v, 0.0, "predict_deriv[{i}] = {v} should be 0");
    }
}

#[test]
fn re_predict_repeats_for_repeated_levels() {
    let (x, y, group_ids) = synth();
    let x2 = x.view().insert_axis(Axis(1));
    let fit = gammon::fit_with_design(
        gammon::family::gaussian_identity(),
        gammon::Re,
        x2,
        y.view(),
        None,
    )
    .expect("fit failed");

    // Predicting at the same level twice produces the same μ̂ (sanity:
    // the basis is genuinely a one-hot lookup, not noisy).
    let probe = Array1::from_vec(vec![group_ids[0], group_ids[0], group_ids[1]]);
    let pred = fit
        .predict(probe.view().insert_axis(Axis(1)))
        .expect("predict failed");
    assert_eq!(
        pred[0], pred[1],
        "two predictions at the same level disagree: {} vs {}",
        pred[0], pred[1]
    );
    assert_ne!(
        pred[0], pred[2],
        "predictions at different levels should differ (group means {} vs {})",
        group_ids[0], group_ids[1]
    );
}
