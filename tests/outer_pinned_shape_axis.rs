//! A shape axis pinned at its bound must not throttle the axes that are still
//! moving.
//!
//! WHAT THE REGIME IS. `scat`'s outer Newton runs over
//! `θ = [ρ, log σ², log(ν − 3)]` inside a box, and `axis_step_caps` bounds each
//! axis's per-iteration movement (5.0 on ρ, 1.0 on each shape axis). Fit scat
//! to data that is NOT heavy-tailed and ν is unidentified: its axis walks to
//! the upper bound of 10 (ν ≈ 22029, i.e. Gaussian), its curvature drops to
//! `|H_ii| ~ 2e-6`, and Newton then asks for +19 on it every single iteration —
//! a step the box clamps away in full.
//!
//! WHAT IT CAUGHT. The cap was applied as ONE global shrink factor
//! `min_i(cap_i/|s_i|)`, computed over the unprojected step, so that clamped-away
//! +19 shrank the whole step by 20×. ρ crawled up its (flat, near-linear-signal)
//! ridge at 0.042 per iteration instead of 0.83, and `log σ²` moved 5e-7 per
//! iteration while carrying a real gradient of 5.6e-2 — FD-confirmed against the
//! score, so it was live curvature being ignored, not noise. The loop burned all
//! 200 iterations without meeting the gradient test and `NotConverged` threw the
//! fit away: `RuntimeError: solver did not converge after 200 iterations
//! (last grad norm = 5.646e-2)` from a single-smooth scat fit that 0.13.1 fit
//! fine. The fix projects the step onto the feasible box before the cap shrink
//! is measured (`outer.rs`, standard projected Newton).
//!
//! Fixture: generated here, deterministic. Shaped after the fit that surfaced
//! this — n = 66 rows, k = 12 (a near-saturated basis, so the ρ ridge is flat),
//! a mild linear signal, Gaussian noise, and a response on a ~1e5 scale.

use ndarray::{Array1, Array2};

/// n = 66, k = 12, near-linear signal, light-tailed noise on a 1e5 response
/// scale. Deterministic Box-Muller off a fixed LCG.
fn fixture() -> (Array2<f64>, Array1<f64>) {
    let n = 66usize;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut lcg = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let x = -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0);
        let u1: f64 = lcg().max(1e-12);
        let u2: f64 = lcg();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        xs.push(x);
        ys.push(550_000.0 + 40_000.0 * x + 20_000.0 * z);
    }
    (
        Array2::from_shape_vec((n, 1), xs).unwrap(),
        Array1::from_vec(ys),
    )
}

#[test]
fn scat_pinned_shape_axis_does_not_throttle_rho() {
    let (x, y) = fixture();
    let mean = y.iter().sum::<f64>() / (y.len() as f64);
    let y_var = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (y.len() as f64);

    let fit = gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y.view(),
        None,
        12,
    )
    .expect("scat with an unidentified ν must still return a fit");

    // shape_params = [log σ², log(ν − MIN_DF)]; ν's axis bound is 10.
    let log_nu_axis = fit.shape_params[1];
    println!(
        "[pinned axis] converged={} iters={} edf={:.4} log(nu-3)={:.4}",
        fit.converged, fit.n_iters, fit.edf_total, log_nu_axis
    );
    assert!(
        log_nu_axis > 9.9,
        "fixture no longer exercises the regime — ν must saturate at its upper \
         bound for the pinned-axis throttle to be reachable; got log(ν−3) = {log_nu_axis:.4}"
    );
    assert!(fit.converged, "the outer Newton must report convergence");
    // The throttled crawl needed all 200 and still missed; the projected step
    // gets there in ~30. Bar set well clear of both.
    assert!(
        fit.n_iters < 80,
        "outer Newton took {} iterations — the pinned ν axis is throttling the \
         live axes again",
        fit.n_iters
    );

    // The fit itself has to be usable, not just non-erroring: a near-linear
    // signal on a flat ρ ridge, so edf should sit near the 2 of the null space
    // and the curve should track the generating line.
    assert!(
        fit.edf_total > 1.5 && fit.edf_total < 6.0,
        "edf {} is not a near-linear fit",
        fit.edf_total
    );
    let pred = fit.predict(x.view()).expect("predict should not fail");
    let max_rel = pred
        .iter()
        .zip(x.column(0).iter())
        .map(|(&p, &xi)| (p - (550_000.0 + 40_000.0 * xi)).abs() / 550_000.0)
        .fold(0.0_f64, f64::max);
    println!("[pinned axis] max rel dev from the generating line: {max_rel:.3e}");
    assert!(
        max_rel < 0.05,
        "fitted curve is {max_rel:.3e} off the generating line"
    );
}
