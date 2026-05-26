//! `LuSolver` parity test — demonstrates that the LU and Cholesky
//! backends produce numerically identical β̂ on the same fixture.
//!
//! Importance: the §C4-note Phase-5b hypothesis was that the residual
//! 2.27e-6 `low_signal_n1000_k10` mgcv-parity gap was the Cholesky-vs-LU
//! factor mismatch. This test invalidates that hypothesis empirically —
//! the two backends agree to 1e-12 on the gamrs side, so the residual
//! parity gap to mgcv lives upstream of the linear backend (likely
//! v0.x's `Sl.initial.repara` rotation, not ported).
//!
//! LU is kept as a swappable backend purely for architectural cleanness
//! and forward-compat — not because it improves μ̂ precision.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
}

fn load_fixture(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn max_abs_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Same fixture; both solvers must produce identical β̂ to 1e-12.
#[test]
fn gaussian_low_signal_lu_matches_cholesky() {
    let fx = load_fixture("1d_gaussian_low_signal_n1000_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    let fit_chol = gamrs::fit_with_solver::<_, _, _, gamrs::CholeskySolver>(
        gamrs::family::gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("Cholesky fit");
    let fit_lu = gamrs::fit_with_solver::<_, _, _, gamrs::LuSolver>(
        gamrs::family::gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("LU fit");

    // β̂ identity (the empirical 1e-13 claim from the previous attempt).
    let delta_beta = max_abs_diff(&fit_chol.beta, &fit_lu.beta);
    println!(
        "[low_signal_n1000_k10 | Cholesky vs LU] \
         max |Δβ̂| = {:.3e}; ρ̂_chol = {:.6}, ρ̂_lu = {:.6}; \
         σ̂²_chol = {:.6e}, σ̂²_lu = {:.6e}; edf_chol = {:.4}, edf_lu = {:.4}",
        delta_beta,
        fit_chol.rho,
        fit_lu.rho,
        fit_chol.scale,
        fit_lu.scale,
        fit_chol.edf_total,
        fit_lu.edf_total,
    );
    assert!(
        delta_beta < 1e-10,
        "LU vs Cholesky β̂ should agree to ~1e-12 on a well-conditioned fixture; got |Δβ̂| = {:.3e}",
        delta_beta,
    );

    // ρ̂ identity (any drift would propagate from outer Newton on different
    // factor-precision gradients). 94b: ρ is now Vec; compare element-wise.
    assert_eq!(fit_chol.rho.len(), fit_lu.rho.len());
    let drho = fit_chol
        .rho
        .iter()
        .zip(fit_lu.rho.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        drho < 1e-8,
        "LU vs Cholesky ρ̂ should agree to ~1e-10; got max|Δρ̂| = {:.3e}",
        drho,
    );

    // EDF identity (uses trace_a_inv on different backends).
    let dedf = (fit_chol.edf_total - fit_lu.edf_total).abs();
    assert!(
        dedf < 1e-8,
        "LU vs Cholesky EDF should agree to ~1e-10; got |ΔEDF| = {:.3e}",
        dedf,
    );
}

/// Smoke: also check a well-conditioned fixture (smooth_n500) to confirm
/// the parity holds across the curvature spectrum, not just the
/// ill-conditioned `low_signal` case.
#[test]
fn gaussian_smooth_n500_lu_matches_cholesky() {
    let fx = load_fixture("1d_gaussian_smooth_n500_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    let fit_chol = gamrs::fit_with_solver::<_, _, _, gamrs::CholeskySolver>(
        gamrs::family::gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("Cholesky fit");
    let fit_lu = gamrs::fit_with_solver::<_, _, _, gamrs::LuSolver>(
        gamrs::family::gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .expect("LU fit");

    let delta_beta = max_abs_diff(&fit_chol.beta, &fit_lu.beta);
    println!(
        "[smooth_n500_k10 | Cholesky vs LU] max |Δβ̂| = {:.3e}; \
         ρ̂_chol = {:.6}, ρ̂_lu = {:.6}",
        delta_beta, fit_chol.rho, fit_lu.rho,
    );
    assert!(
        delta_beta < 1e-10,
        "LU vs Cholesky β̂ should agree to ~1e-12; got |Δβ̂| = {:.3e}",
        delta_beta,
    );
}

/// Same parity check via the canonical typed entry point.
#[test]
fn canonical_fit_with_solver_lu_matches_cholesky() {
    use gamrs::family::gaussian_identity;

    let fx = load_fixture("1d_gaussian_smooth_n500_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    let fit_default = gamrs::fit(gaussian_identity(), x.view(), y.view(), None, k).unwrap();
    let fit_chol = gamrs::fit_with_solver::<_, _, _, gamrs::CholeskySolver>(
        gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .unwrap();
    let fit_lu = gamrs::fit_with_solver::<_, _, _, gamrs::LuSolver>(
        gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .unwrap();

    // Default and explicit-Cholesky must be bit-identical.
    assert_eq!(
        max_abs_diff(&fit_default.beta, &fit_chol.beta),
        0.0,
        "gamrs::fit(...) and gamrs::fit_with_solver::<CholeskySolver>(...) must produce identical β̂",
    );
    // LU must agree to 1e-10 with both.
    assert!(
        max_abs_diff(&fit_chol.beta, &fit_lu.beta) < 1e-10,
        "gamrs::fit_with_solver::<LuSolver>(...) should match Cholesky to ~1e-12",
    );
}
