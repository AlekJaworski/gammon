//! **Two public entry points must pose the same problem.**
//!
//! `gamrs::fit(family, x, y, w, k)` goes straight to `Cr { k }`.
//! `gamrs::fit_with_design(family, Additive { terms: [Cr { col, k }] }, …)` is
//! what the Python `Gam` path builds. Same basis, same k, same data — so they
//! must agree, and if they do not, every measurement is entry-point dependent
//! and no parity claim means anything until you say which door you came in by.
//!
//! This exists because the 2026-08 scat investigation spent a long time calling
//! a discrepancy "start-sensitivity" when the two numbers being compared
//! (max_rel 9.456e-4 and 1.095e-4 on the same fixture) came from these two
//! entry points, not from two starts. Fixture is fully synthetic.

use gamrs::design::{Additive, TermSpec};
use gamrs::family::tdist_identity;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fx {
    inputs: In,
    unique_x: Vec<f64>,
    mgcv_output: Arms,
}
#[derive(Deserialize)]
struct In {
    x_train: Vec<f64>,
    y_train: Vec<f64>,
    k: usize,
}
#[derive(Deserialize)]
struct Arms {
    gam_reml: Arm,
}
#[derive(Deserialize)]
struct Arm {
    edf: f64,
    nu: f64,
    predictions_unique_x: Vec<f64>,
}

fn load() -> Fx {
    let p: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests/fixtures/1d_scat_saturated_basis_n620_k5_cr.json",
    ]
    .iter()
    .collect();
    serde_json::from_str(&std::fs::read_to_string(p).expect("fixture")).expect("parse")
}

fn max_rel(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, t)| (p - t).abs() / t.abs().max(1e-12))
        .fold(0.0_f64, f64::max)
}

fn sd_of(y: &Array1<f64>) -> f64 {
    let n = y.len() as f64;
    let m = y.iter().sum::<f64>() / n;
    (y.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// (x, y, k, unique_x) from the synthetic saturated-basis fixture.
fn case() -> (Array2<f64>, Array1<f64>, usize, Array2<f64>) {
    let fx = load();
    let n = fx.inputs.y_train.len();
    (
        Array2::from_shape_vec((n, 1), fx.inputs.x_train.clone()).unwrap(),
        Array1::from_vec(fx.inputs.y_train.clone()),
        fx.inputs.k,
        Array2::from_shape_vec((fx.unique_x.len(), 1), fx.unique_x.clone()).unwrap(),
    )
}

fn fit_at(x: &Array2<f64>, y: &Array1<f64>, k: usize, nu: f64, sigma2: f64) -> gamrs::FittedGam {
    gamrs::fit(tdist_identity(nu, sigma2), x.view(), y.view(), None, k)
        .unwrap_or_else(|e| panic!("scat fit failed at nu={nu} sigma2={sigma2:.4e}: {e}"))
}

#[test]
fn cr_and_additive_entry_points_agree_on_scat() {
    let fx = load();
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 1), fx.inputs.x_train.clone()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k;
    let ux = Array2::from_shape_vec((fx.unique_x.len(), 1), fx.unique_x.clone()).unwrap();

    // Identical family init on both sides so the ONLY difference is the door.
    let y_var = {
        let m = y.iter().sum::<f64>() / (n as f64);
        y.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (n as f64)
    };
    let init = || tdist_identity(5.0, y_var * 0.1);

    let a = gamrs::fit(init(), x.view(), y.view(), None, k).expect("Cr entry");
    let b = gamrs::fit_with_design(
        init(),
        Additive {
            terms: vec![TermSpec::Cr { col: 0, k }],
        },
        x.view(),
        y.view(),
        None,
    )
    .expect("Additive entry");

    let pa = a.predict(ux.view()).unwrap();
    let pb = b.predict(ux.view()).unwrap();
    let m = &fx.mgcv_output.gam_reml;

    println!(
        "  Cr{{k}}   : edf={:.4}  rho={:?}  iters={}  max_rel(mgcv)={:.3e}",
        a.edf_total,
        a.rho.to_vec(),
        a.n_iters,
        max_rel(pa.as_slice().unwrap(), &m.predictions_unique_x)
    );
    println!(
        "  Additive : edf={:.4}  rho={:?}  iters={}  max_rel(mgcv)={:.3e}",
        b.edf_total,
        b.rho.to_vec(),
        b.n_iters,
        max_rel(pb.as_slice().unwrap(), &m.predictions_unique_x)
    );
    println!("  mgcv     : edf={:.4}  nu={:.4}", m.edf, m.nu);

    let between = max_rel(pa.as_slice().unwrap(), pb.as_slice().unwrap());
    println!("  entry-point disagreement: max_rel = {between:.3e}");
    assert!(
        between < 1.0e-10,
        "the two entry points do not pose the same problem: curves differ by \
         {between:.3e} (Cr edf {:.4} vs Additive edf {:.4}). Until they agree, \
         no parity number is meaningful without naming the entry point.",
        a.edf_total,
        b.edf_total
    );
}

/// **The ν start does not matter.** mgcv's own scat init is
/// `ν = exp(1.5) + min.df ≈ 7.482` (`efam.r:169-177`); gamrs ships ν = 5.
/// Measured difference between the two: ~1e-5 relative on the curve. Bounded,
/// so the ν choice is a cosmetic difference from mgcv, not a load-bearing one.
#[test]
fn scat_fit_is_insensitive_to_the_nu_start() {
    let (x, y, k, ux) = case();
    let sd = sd_of(&y);
    let a = fit_at(&x, &y, k, 5.0, sd * sd);
    let b = fit_at(&x, &y, k, (1.5_f64).exp() + 3.0, (0.8 * sd).powi(2));
    let pa = a.predict(ux.view()).unwrap();
    let pb = b.predict(ux.view()).unwrap();
    let d = max_rel(pa.as_slice().unwrap(), pb.as_slice().unwrap());
    println!(
        "  nu0=5 -> edf {:.4} ({} iters);  nu0=7.482 (mgcv) -> edf {:.4} ({} iters);  \
         curve diff {d:.3e}",
        a.edf_total, a.n_iters, b.edf_total, b.n_iters
    );
    assert!(
        d < 1.0e-4,
        "the nu start became load-bearing: curves differ by {d:.3e} \
         (edf {:.4} vs {:.4})",
        a.edf_total,
        b.edf_total
    );
}

/// **KNOWN DEFECT: the σ² start IS load-bearing, and low variance is the bad
/// side.** mgcv says so out loud — `efam.r:172-174`, "low df and low variance
/// promotes indefiniteness. Better to start with moderate df and fairly high
/// variance" — and starts at `σ = 0.8·sd(y)`. gamrs's own default is
/// `σ² = sd(y)²`, which is on the safe side; but nothing stops a caller passing
/// a low one, and `tests/parity_scat_tf9963.rs` does exactly that.
///
/// Measured on this fixture with the observed-`log|H|` criterion: starting at
/// `σ² = 0.1·sd²` instead of `sd²` moves edf 2.3754 → 2.4262 and the curve
/// error against mgcv 1.1e-4 → 9.5e-4, a 9× degradation from the start alone.
///
/// This test bounds the damage so a fix shows up as improvement and a
/// regression as failure. It is NOT a licence to keep the sensitivity — the
/// fix is to make the outer loop reach the same optimum regardless, and/or to
/// adopt mgcv's init. Do not loosen the bound to make something else pass.
#[test]
fn scat_low_variance_start_degrades_the_fit_by_a_bounded_amount() {
    let (x, y, k, ux) = case();
    let sd = sd_of(&y);
    let good = fit_at(&x, &y, k, 5.0, sd * sd);
    let poor = fit_at(&x, &y, k, 5.0, 0.1 * sd * sd);
    let pg = good.predict(ux.view()).unwrap();
    let pp = poor.predict(ux.view()).unwrap();
    let d = max_rel(pg.as_slice().unwrap(), pp.as_slice().unwrap());
    println!(
        "  s2_0=sd^2 -> edf {:.4};  s2_0=0.1*sd^2 -> edf {:.4};  curve diff {d:.3e}",
        good.edf_total, poor.edf_total
    );
    assert!(
        d < 2.0e-3,
        "the low-variance start got WORSE than the recorded 1e-3: {d:.3e}"
    );
}
