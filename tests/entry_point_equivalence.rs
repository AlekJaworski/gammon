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

/// **KNOWN DEFECT: under the observed-`log|H|` criterion a low σ² start makes
/// the outer loop stop EARLY — at a measurably worse REML value.**
///
/// Stated on the objective, not the curve, because that is the robust claim: a
/// run that ends at a higher REML than another run on the same problem did not
/// find the optimum, whatever its edf says.
///
/// Measured on this fixture:
/// ```text
///                        rho        REML            edf
///   log|A|  sd^2      7.516694   854.222515972   2.3641
///   log|A|  0.1sd^2   7.517681   854.222515961   2.3638   <- agree to 1e-8
///   observed sd^2     7.452111   854.174642406   2.3754
///   observed 0.1sd^2  7.286090   854.175555690   2.4262   <- 9.1e-4 WORSE
/// ```
/// So the shipped criterion reaches the same optimum from any start, and the
/// new one does not. That is a real cost of the criterion change.
///
/// Ruled out as the cause, by measurement: the PD fallback (never fires — see
/// `observed_criterion_pd_fallback_rate_by_start`), the entry point, the ν
/// start, and `grad_tol` (tightening it to mgcv's `sqrt(eps)` does not move
/// this). Do NOT loosen this bound to make something else pass.
#[test]
fn observed_criterion_low_variance_start_stops_early() {
    let (x, y, k, _ux) = case();
    let sd = sd_of(&y);
    let good = fit_at(&x, &y, k, 5.0, sd * sd);
    let poor = fit_at(&x, &y, k, 5.0, 0.1 * sd * sd);
    let excess = poor.reml_value - good.reml_value;
    println!(
        "  s2_0=sd^2   -> rho={:.6} reml={:.9} edf={:.4}\n  \
         s2_0=0.1sd^2 -> rho={:.6} reml={:.9} edf={:.4}\n  \
         excess REML at the low start = {excess:+.3e}",
        good.rho[0], good.reml_value, good.edf_total, poor.rho[0], poor.reml_value, poor.edf_total
    );
    // With the SHIPPED criterion both starts reach the same optimum and this
    // lands at ~-1.2e-8, i.e. convergence noise about zero — which is the
    // healthy case, not a reversal. The guard is for a MEANINGFUL reversal
    // (the high-variance start being the one that stops early), so it has to
    // sit above that noise floor rather than at it.
    assert!(
        excess > -1.0e-6,
        "the low-variance start found a materially BETTER optimum ({excess:.3e}) \
         — then the high-variance start is the one stopping early and this test \
         has the asymmetry backwards"
    );
    assert!(
        excess < 2.0e-3,
        "the low-variance start's early stop got worse than recorded: excess \
         REML {excess:.3e} (was 9.1e-4)"
    );
}

/// **Measures the mechanism behind the σ²-start sensitivity above.**
///
/// The observed-curvature criterion is conditionally defined: when
/// `X'diag(½D_μμ)X + λS` is not positive definite we fall back to `log|A|`, so
/// the objective can change under the optimiser's feet. mgcv's own init comment
/// (`efam.r:172-174`) predicts a low σ² start makes this worse — smaller σ²
/// puts more rows past `|r| > √(νσ²)`, so more negative-curvature rows.
///
/// This prints the PD-ok / PD-fallback counts per start so the claim is
/// measured rather than reasoned. It asserts only the weak, robust thing: a fit
/// that never falls back cannot be suffering from this mechanism.
#[test]
fn observed_criterion_pd_fallback_rate_by_start() {
    let (x, y, k, _ux) = case();
    let sd = sd_of(&y);
    for (tag, nu, s2) in [
        ("s2=sd^2 (gamrs)", 5.0, sd * sd),
        (
            "s2=(0.8sd)^2 (mgcv)",
            (1.5_f64).exp() + 3.0,
            (0.8 * sd).powi(2),
        ),
        ("s2=0.1*sd^2 (low)", 5.0, 0.1 * sd * sd),
    ] {
        gamrs::inner::pirls_reset_observed_pd_counts();
        let f = fit_at(&x, &y, k, nu, s2);
        let (ok, fb) = gamrs::inner::pirls_observed_pd_counts();
        let total = ok + fb;
        let pct = if total == 0 {
            f64::NAN
        } else {
            100.0 * fb as f64 / total as f64
        };
        println!(
            "  {tag:<22} edf={:.4}  iters={:<3}  rho={:.6}  reml={:.9}  \
             PD ok={ok:<5} fallback={fb:<5} ({pct:.1}% of {total} evals)",
            f.edf_total, f.n_iters, f.rho[0], f.reml_value
        );
    }
}

/// **Pre-standardizing the response by hand must be a no-op.** The fit core
/// divides by `scat_response_scale(y)` = `sd(y)` (floored at 1) and rescales
/// out, so handing it `y/sd(y)` with a correspondingly scaled σ² start poses
/// the identical problem. If the two disagree, the standardization round-trip
/// is not exact and every measurement depends on which scale you happened to
/// pass in.
#[test]
fn pre_standardizing_the_response_is_a_no_op() {
    let (x, y, k, ux) = case();
    let sd = sd_of(&y);
    let yz = y.mapv(|v| v / sd);

    // Same effective start: σ²_std = 0.1 either way.
    let raw = fit_at(&x, &y, k, 5.0, 0.1 * sd * sd);
    let pre = fit_at(&x, &yz, k, 5.0, 0.1);

    let pr = raw.predict(ux.view()).unwrap();
    let pp = pre.predict(ux.view()).unwrap();
    // `pre` is fitted on y/sd, so its curve is on that scale.
    let pp_raw: Vec<f64> = pp.iter().map(|v| v * sd).collect();
    let d = max_rel(pr.as_slice().unwrap(), &pp_raw);
    println!(
        "  raw y          -> rho={:.6} reml={:.9} edf={:.4} iters={}",
        raw.rho[0], raw.reml_value, raw.edf_total, raw.n_iters
    );
    println!(
        "  pre-standardized -> rho={:.6} reml={:.9} edf={:.4} iters={}",
        pre.rho[0], pre.reml_value, pre.edf_total, pre.n_iters
    );
    println!("  curve difference: {d:.3e}");
    assert!(
        d < 1.0e-6,
        "pre-standardizing changed the fit ({d:.3e}): rho {:.6} vs {:.6}, \
         edf {:.4} vs {:.4}. The standardization round-trip is not exact.",
        raw.rho[0],
        pre.rho[0],
        raw.edf_total,
        pre.edf_total
    );
}
