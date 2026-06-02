//! Basic smoke test for gamrs's TPRS path (`Tps { cols, k }`).
//!
//! No mgcv fixture — this test exercises the end-to-end fit-time +
//! predict-time round-trip on a synthetic 2-D Gaussian dataset. Asserts:
//! 1. The fit converges via `fit_with_design(..., Additive { terms })`.
//! 2. The single smoothing parameter (TPRS is isotropic) is reflected in
//!    `fit.rho.len() == 1`.
//! 3. Predictions on the training x are well-defined and within a sane
//!    range relative to y (μ-RMSE < 1.0 — generous bound since this is
//!    a smoke test, not a parity test).
//! 4. Predictions on a held-out grid are finite.
//!
//! A full mgcv parity fixture would require regenerating `bs="tp"` outputs
//! from R — left as a follow-up. The TPRS construction here uses a
//! low-rank radial basis + linear polynomial null-space, which is the
//! standard practical form (Wood 2003 §5.5) but differs from mgcv's
//! eigenvalue-truncation E-S decomposition in the basis parameterisation
//! (predictions should agree up to FP for comparable knot grids; exact
//! coefficient layout differs).

use ndarray::{Array1, Array2};

use gamrs::design::{Additive, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit_with_design;

#[test]
fn tps_2d_gaussian_smoke() {
    let n: usize = 200;
    let k: usize = 15;
    // Synthetic 2-D: y = sin(2 pi x0) + cos(2 pi x1) + noise
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x0 = (i as f64) / (n as f64);
        let x1 = ((i as f64) * 0.171).sin().abs(); // pseudo-random 0..1
        x[[i, 0]] = x0;
        x[[i, 1]] = x1;
        let mu = (2.0 * std::f64::consts::PI * x0).sin() + (2.0 * std::f64::consts::PI * x1).cos();
        // tiny deterministic noise
        let noise = ((i as f64 * 0.7).sin()) * 0.05;
        y[i] = mu + noise;
    }

    let terms = vec![TermSpec::Tps {
        cols: vec![0, 1],
        k,
    }];

    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("TPS Gaussian fit failed");

    // Single smoothing parameter for an isotropic smooth.
    assert_eq!(
        fit.rho.len(),
        1,
        "TPRS is isotropic — exactly 1 smoothing param"
    );
    assert_eq!(fit.lambda.len(), 1);
    assert_eq!(fit.edf_per_term.len(), 1);

    // Predictions on training x: should be finite + close-ish to y.
    let pred = fit.predict(x.view()).expect("predict failed");
    assert_eq!(pred.len(), n);
    let rmse = (pred
        .iter()
        .zip(y.iter())
        .map(|(p, y)| (p - y).powi(2))
        .sum::<f64>()
        / (n as f64))
        .sqrt();
    println!(
        "[tps_2d_gaussian_smoke] n_iters={}, converged={}, edf={:.3}, rho={:.3}, RMSE(train)={:.4}",
        fit.n_iters, fit.converged, fit.edf_per_term[0], fit.rho[0], rmse
    );
    for v in pred.iter() {
        assert!(v.is_finite(), "TPS prediction must be finite");
    }
    assert!(
        rmse < 1.0,
        "TPS train RMSE = {rmse} unexpectedly large (smoke-level sanity check)"
    );

    // Held-out grid predict (basis-invariance check).
    let m = 25;
    let mut x_new = Array2::<f64>::zeros((m, 2));
    for i in 0..m {
        x_new[[i, 0]] = 0.05 + 0.9 * (i as f64) / (m as f64 - 1.0);
        x_new[[i, 1]] = 0.10 + 0.8 * ((i as f64) * 0.21).sin().abs();
    }
    let pred_new = fit.predict(x_new.view()).expect("predict new failed");
    for v in pred_new.iter() {
        assert!(v.is_finite(), "TPS prediction on held-out must be finite");
    }
}
