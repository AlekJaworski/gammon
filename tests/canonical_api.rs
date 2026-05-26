//! Canonical entry-point smoke tests: `gammon::fit(family, x, y, w, k)`
//! must converge and produce sane output on Gaussian, Bernoulli,
//! Tweedie, and Ocat — i.e. the type-driven dispatch reaches the right
//! driver. Per-family parity vs mgcv lives in the dedicated `parity_*`
//! suites; here we just exercise the dispatch surface.

use ndarray::{Array1, Axis};

use gammon::family::{bernoulli_logit, gaussian_identity, ocat_identity, tweedie_log};

fn logistic(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

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
fn canonical_fit_gaussian_converges() {
    let n = 200;
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, n);
    let y: Array1<f64> = x.iter().map(|&xi| (2.0 * xi).sin()).collect();

    let canon = gammon::fit(
        gaussian_identity(),
        x.view().insert_axis(Axis(1)),
        y.view(),
        None,
        10,
    )
    .expect("gammon::fit Gaussian should not fail");

    assert!(canon.converged);
    assert!(canon.scale.is_finite() && canon.scale > 0.0);
    assert!(canon.edf_total.is_finite() && canon.edf_total > 0.0);
}

#[test]
fn canonical_fit_bernoulli_converges() {
    let n = 300;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let eta = 4.0 * (x - 0.5);
            pseudo_bernoulli(i, logistic(eta))
        })
        .collect();
    let x = Array1::from_vec(xs);
    let y = Array1::from_vec(ys);

    let canon = gammon::fit(
        bernoulli_logit(),
        x.view().insert_axis(Axis(1)),
        y.view(),
        None,
        10,
    )
    .expect("gammon::fit Bernoulli should not fail");

    assert_eq!(canon.scale, 1.0, "Bernoulli σ² fixed at 1");
    assert!(canon.edf_total.is_finite() && canon.edf_total > 0.0);
}

#[test]
fn canonical_fit_tweedie_dispatches_to_shape_aware_driver() {
    // Tweedie's shape params (p, φ) live on the family — canonical API
    // reads them off `family.loss.p` / `family.loss.phi` and dispatches
    // to the shape-aware joint-Newton driver. Smoke-check the surface.
    let n = 250;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
    // Tweedie-ish data: mostly 0, occasional positive bursts.
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let h = (i.wrapping_mul(2654435761)) as u32;
            let u = (h as f64) / (u32::MAX as f64);
            let mu = (2.0 * x).exp(); // ~1 to ~7.4
            if u < 0.3 {
                0.0
            } else {
                mu * (1.0 + 0.5 * (u - 0.5))
            }
        })
        .collect();
    let x = Array1::from_vec(xs);
    let y = Array1::from_vec(ys);

    let init_p = 1.5;
    let init_phi = 1.0;
    let canon = gammon::fit(
        tweedie_log(init_p, init_phi),
        x.view().insert_axis(Axis(1)),
        y.view(),
        None,
        8,
    )
    .expect("gammon::fit Tweedie should not fail");

    // scale = φ̂ (Tweedie convention).
    assert!(canon.scale.is_finite() && canon.scale > 0.0);
    assert!(canon.edf_total.is_finite() && canon.edf_total > 0.0);
}

#[test]
fn canonical_fit_ocat_dispatches() {
    // Ocat carries n_cats + thresholds on the family — canonical entry
    // unpacks them and dispatches into the shape-aware Ocat driver.
    let n = 200;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| {
            // 3 categories driven by x with sharp transitions.
            if x < 0.33 {
                1.0
            } else if x < 0.66 {
                2.0
            } else {
                3.0
            }
        })
        .collect();
    let x = Array1::from_vec(xs);
    let y = Array1::from_vec(ys);

    let n_cats = 3;
    let theta0 = Array1::<f64>::zeros(n_cats - 2);
    let canon = gammon::fit(
        ocat_identity(theta0, n_cats),
        x.view().insert_axis(Axis(1)),
        y.view(),
        None,
        8,
    )
    .expect("gammon::fit Ocat should not fail");

    assert_eq!(canon.scale, 1.0, "Ocat dispersion is fixed at 1");
    assert!(canon.edf_total.is_finite() && canon.edf_total > 0.0);
}

// ---------------------------------------------------------------------------
// Error-message ergonomics — the canonical API surface MUST emit
// actionable, location-bearing errors (per the project ergonomic
// directive). Each guard is exercised through the wrappers (which is
// where `gammon::fit` ends up after dispatch).
// ---------------------------------------------------------------------------

#[test]
fn error_bernoulli_y_out_of_unit_carries_row() {
    let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let y = Array1::from_vec(vec![0.0, 1.0, 1.5, 0.0]); // bad at row 2
    let err = match gammon::fit(
        bernoulli_logit(),
        x.view().insert_axis(Axis(1)),
        y.view(),
        None,
        4,
    ) {
        Ok(_) => panic!("Bernoulli with y=1.5 must error"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("row 2"),
        "expected row-index in error, got: {msg}"
    );
    assert!(
        msg.contains("Bernoulli"),
        "expected family name in error, got: {msg}"
    );
}

#[test]
fn error_quantile_tau_out_of_range_is_actionable() {
    // ELF family with tau=1.0 (boundary) must reject with a message that
    // names the constraint AND the offending value.
    let x = Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8]);
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let err = match gammon::fit(
        gammon::family::elf_identity(1.0, -1.0, -1.0),
        x.view().insert_axis(Axis(1)),
        y.view(),
        None,
        4,
    ) {
        Ok(_) => panic!("tau=1.0 must error"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("(0, 1)"),
        "expected open-interval in error, got: {msg}"
    );
    assert!(
        msg.contains("tau=1"),
        "expected offending tau in error, got: {msg}"
    );
}
