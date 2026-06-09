//! Extrapolation behaviour for CR smooths.
//!
//! mgcv's cubic-regression splines extrapolate **linearly** beyond the boundary
//! knots, and that continuation happens on the **link (η) scale** — so:
//!   - identity link  ⟹ μ continues linearly,
//!   - log link        ⟹ μ continues exponentially but stays > 0,
//!   - logit link      ⟹ μ saturates within (0, 1).
//!
//! The first derivative is constant outside the data range (the boundary
//! slope), and predictions never go NaN/inf however far out you ask.
//!
//! The linear-in-η property is a feature of the CR basis itself (the design
//! rows are `value + slope·dx` beyond the range, see `basis/cr.rs`), so it
//! holds for any fitted β — these tests lock the behaviour in across the link
//! families and guard against a future basis change silently breaking it.

use std::f64::consts::PI;

use ndarray::{Array1, Array2};

fn grid(xs: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((xs.len(), 1), xs.to_vec()).unwrap()
}

/// Equally-spaced inputs ⟹ a linear response has ~zero second differences.
fn assert_linear(seq: &[f64], ctx: &str) {
    assert!(seq.len() >= 3, "{ctx}: need ≥3 points");
    let step = seq[1] - seq[0];
    let max2 = seq
        .windows(3)
        .map(|w| (w[2] - 2.0 * w[1] + w[0]).abs())
        .fold(0.0_f64, f64::max);
    let rel = max2 / (step.abs() + 1e-12);
    assert!(
        rel < 1e-6,
        "{ctx}: not linear beyond range — max|2nd diff|={max2:.3e}, rel-to-step={rel:.3e}"
    );
}

fn assert_all_finite(seq: &[f64], ctx: &str) {
    assert!(
        seq.iter().all(|v| v.is_finite()),
        "{ctx}: non-finite prediction(s): {seq:?}"
    );
}

// Equally-spaced probes beyond a [0,1] training range (ascending, so second
// differences are meaningful), plus an arbitrarily-far set for the NaN guard.
const HI: [f64; 5] = [1.5, 2.0, 2.5, 3.0, 3.5];
const LO: [f64; 5] = [-2.5, -2.0, -1.5, -1.0, -0.5];
const FAR: [f64; 4] = [-1.0e3, -50.0, 50.0, 1.0e6];

fn unit_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect()
}

#[test]
fn cr_identity_extrapolates_linearly_both_sides() {
    let xs = unit_grid(100);
    let ys: Vec<f64> = xs.iter().map(|&x| (2.0 * PI * x).sin()).collect();
    let fit = gamrs::fit(
        gamrs::family::gaussian_identity(),
        grid(&xs).view(),
        Array1::from(ys).view(),
        None,
        10,
    )
    .expect("gaussian fit");

    // Identity link: predict() IS μ. Both tails must be linear.
    let hi = fit.predict(grid(&HI).view()).unwrap();
    let lo = fit.predict(grid(&LO).view()).unwrap();
    assert_linear(hi.as_slice().unwrap(), "identity hi-extrap");
    assert_linear(lo.as_slice().unwrap(), "identity lo-extrap");

    // Derivative is the (constant) boundary slope beyond the range.
    let d_hi = fit.predict_deriv(grid(&HI).view()).unwrap();
    let span = d_hi.iter().cloned().fold(f64::MIN, f64::max)
        - d_hi.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        span.abs() < 1e-6,
        "predict_deriv not constant beyond range: span={span:.3e}"
    );

    // Never NaN/inf, however far out.
    let far = fit.predict(grid(&FAR).view()).unwrap();
    assert_all_finite(far.as_slice().unwrap(), "identity far");
}

#[test]
fn cr_log_link_extrapolation_linear_in_eta_positive_mu() {
    let xs = unit_grid(80);
    // Deterministic positive counts; exact shape is irrelevant (linearity is
    // a basis property), the fit just needs to converge.
    let ys: Vec<f64> = xs.iter().map(|&x| (0.5 + 0.8 * x).exp().round()).collect();
    let fit = gamrs::fit(
        gamrs::family::poisson_log(),
        grid(&xs).view(),
        Array1::from(ys).view(),
        None,
        8,
    )
    .expect("poisson fit");

    // predict() is η (log μ): must be linear beyond range.
    let eta_hi = fit.predict(grid(&HI).view()).unwrap();
    let eta_lo = fit.predict(grid(&LO).view()).unwrap();
    assert_linear(eta_hi.as_slice().unwrap(), "log η hi-extrap");
    assert_linear(eta_lo.as_slice().unwrap(), "log η lo-extrap");

    // Moderate extrapolation: μ = exp(η) is finite and strictly positive.
    for xset in [&HI[..], &LO[..]] {
        let eta = fit.predict(grid(xset).view()).unwrap();
        let mu: Vec<f64> = eta.iter().map(|&e| fit.link_kind.inverse(e)).collect();
        assert_all_finite(&mu, "log μ moderate");
        assert!(
            mu.iter().all(|&m| m > 0.0),
            "log link μ must stay > 0: {mu:?}"
        );
    }
    // Extreme far-out: exp(η) may overflow to +inf, but the link stays
    // well-behaved — never NaN, never negative (monotone, sign-correct). This
    // graceful overflow (vs a garbage value) is the behaviour we want to keep.
    let mu_far: Vec<f64> = fit
        .predict(grid(&FAR).view())
        .unwrap()
        .iter()
        .map(|&e| fit.link_kind.inverse(e))
        .collect();
    assert!(
        mu_far.iter().all(|&m| !m.is_nan() && m >= 0.0),
        "log μ far must be ≥0 / non-NaN (may be +inf): {mu_far:?}"
    );
}

#[test]
fn cr_logit_link_extrapolation_linear_in_eta_saturates_unit_interval() {
    let xs = unit_grid(120);
    // Non-separable Bernoulli-ish labels with prob increasing in x (golden-
    // ratio jitter avoids perfect separation, which would diverge the fit).
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let u = ((i as f64) * 0.618_033_988_75).fract();
            if u < 0.2 + 0.6 * x {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let fit = gamrs::fit(
        gamrs::family::bernoulli_logit(),
        grid(&xs).view(),
        Array1::from(ys).view(),
        None,
        8,
    )
    .expect("bernoulli fit");

    // η linear beyond range; μ = logit⁻¹(η) saturates inside (0,1).
    let eta_hi = fit.predict(grid(&HI).view()).unwrap();
    assert_linear(eta_hi.as_slice().unwrap(), "logit η hi-extrap");

    for xset in [&HI[..], &LO[..], &FAR[..]] {
        let eta = fit.predict(grid(xset).view()).unwrap();
        let mu: Vec<f64> = eta.iter().map(|&e| fit.link_kind.inverse(e)).collect();
        assert_all_finite(&mu, "logit μ");
        assert!(
            mu.iter().all(|&m| (0.0..=1.0).contains(&m)),
            "logit μ must stay in [0,1]: {mu:?}"
        );
    }
}
