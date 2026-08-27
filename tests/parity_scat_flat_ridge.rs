//! `scat` on a REML surface with NO interior λ optimum — the regime behind the
//! open disagreement in `docs/scat_adjuster_parity_2026-08.md`.
//!
//! WHAT THE REGIME IS. When the signal is (near-)linear in a coarse ordinal
//! predictor, the REML score of a CR smooth has no interior minimum in λ: it
//! falls monotonically toward the λ→∞ null-space limit, where the smooth
//! collapses to a straight line and edf → 2. The descent is slow — the
//! fixture's own `sp_ladder` records two decades of `sp` costing mgcv 0.0024
//! REML units — but it is a descent, and the fitted curve moves hundreds of
//! dollars across it. mgcv's outer Newton walks it to the limit; gamrs's stops
//! part-way.
//!
//! WHY IT IS NOT the fixture in `parity_scat_tf9963.rs`. That one locks a term
//! with an interior optimum (`garage_spaces`, edf ≈ 2.42) and catches a
//! *direction-of-penalty* defect. This one has no interior optimum at all, and
//! catches an *early-stop* defect: the two failure modes need different data.
//!
//! WHAT IS STILL OPEN. gamrs lands at edf ≈ 2.17 against mgcv's 2.010 — 0.16
//! spurious degrees of freedom, ~$370 on a $548k curve. Its own REML score is
//! still falling in λ at that point (measured on the real terms in
//! `docs/scat_adjuster_parity_2026-08.md`), so this is gamrs stopping short of
//! its own optimum, not a difference of criterion. The first test below pins
//! the gap so it cannot silently widen; the second states the target and is
//! `#[ignore]`d until `src/outer.rs`'s step-halving fallback is fixed.
//!
//! Fixture: fully synthetic — `scripts/r/gen_scat_flat_ridge_fixture.R`, seed 1.
//! No customer data (the real housing residuals stay in the gitignored `data/`).

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    unique_x: Vec<f64>,
    mgcv_output: MgcvOutput,
    sp_ladder: Vec<Rung>,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<f64>,
    y_train: Vec<f64>,
    k: usize,
}

#[derive(Deserialize)]
struct MgcvOutput {
    gam_reml: Arm,
}

#[derive(Deserialize)]
struct Arm {
    sp: f64,
    edf: f64,
    nu: f64,
    sigma: f64,
    reml: f64,
    predictions_unique_x: Vec<f64>,
}

#[derive(Deserialize)]
struct Rung {
    sp: f64,
    reml: f64,
    edf: f64,
}

fn load() -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/1d_scat_flat_lambda_ridge_n620_k12_cr.json");
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

struct Fitted {
    edf_total: f64,
    nu: f64,
    max_abs_dollars: f64,
    max_rel: f64,
    converged: bool,
    n_iters: usize,
}

fn fit_gamrs(fx: &Fixture) -> Fitted {
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 1), fx.inputs.x_train.clone()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let mean = y.iter().sum::<f64>() / (n as f64);
    let y_var = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64);

    let fit = gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y.view(),
        None,
        fx.inputs.k,
    )
    .expect("gamrs::fit (scat) should not fail");

    let ux = Array2::from_shape_vec((fx.unique_x.len(), 1), fx.unique_x.clone()).unwrap();
    let pred = fit.predict(ux.view()).expect("predict should not fail");
    let target = &fx.mgcv_output.gam_reml.predictions_unique_x;
    let (mut max_abs, mut max_rel) = (0.0_f64, 0.0_f64);
    for (p, t) in pred.iter().zip(target) {
        max_abs = max_abs.max((p - t).abs());
        max_rel = max_rel.max((p - t).abs() / t.abs());
    }
    // shape_params = [log σ², log(ν − MIN_DF)]; MIN_DF = 3 (mgcv scat min.df).
    let nu = 3.0 + fit.shape_params[1].exp();
    Fitted {
        edf_total: fit.edf_total,
        nu,
        max_abs_dollars: max_abs,
        max_rel,
        converged: fit.converged,
        n_iters: fit.n_iters,
    }
}

fn report(fx: &Fixture, g: &Fitted) {
    let m = &fx.mgcv_output.gam_reml;
    println!(
        "[scat flat ridge] mgcv: sp={:.6e} edf={:.4} nu={:.4} sigma={:.1} reml={:.4}",
        m.sp, m.edf, m.nu, m.sigma, m.reml
    );
    println!(
        "                  gamrs: edf={:.4} nu={:.4} iters={} converged={}",
        g.edf_total, g.nu, g.n_iters, g.converged
    );
    println!(
        "                  curve gap: ${:.1} max abs, {:.3e} max rel; edf gap {:+.4}",
        g.max_abs_dollars,
        g.max_rel,
        g.edf_total - m.edf
    );
    println!("                  mgcv's own sp ladder (the ridge this test is about):");
    for r in &fx.sp_ladder {
        println!(
            "                    sp={:<12.4e} reml={:>12.4} edf={:.4}",
            r.sp, r.reml, r.edf
        );
    }
}

/// The ridge really has no interior optimum — mgcv's own score keeps falling as
/// `sp` grows past its reported optimum. If this ever stops holding, the fixture
/// has drifted into a different regime and the test below is testing nothing.
#[test]
fn flat_ridge_fixture_really_has_no_interior_lambda_optimum() {
    let fx = load();
    let last = fx.sp_ladder.last().expect("ladder must be non-empty");
    let at_opt = &fx.mgcv_output.gam_reml;
    assert!(
        last.edf < 2.01,
        "the top of the sp ladder should be at the straight-line limit; edf {:.4}",
        last.edf
    );
    assert!(
        last.reml <= at_opt.reml,
        "REML at the top of the ladder ({:.6}) should not be worse than at mgcv's \
         reported optimum ({:.6}) — this fixture is meant to have NO interior optimum",
        last.reml,
        at_opt.reml
    );
    // …and the descent is shallow: that is what lets an optimiser stop early.
    assert!(
        (at_opt.reml - last.reml).abs() < 0.05,
        "ridge should be flat to well under a REML unit; got {:.6}",
        at_opt.reml - last.reml
    );
}

/// Regression guard on the OPEN gap. Not a parity assertion — gamrs does not
/// match mgcv here. It pins ν and the fitted level (which do agree) and brackets
/// the edf overshoot so it can neither widen nor silently vanish unnoticed.
#[test]
fn scat_flat_lambda_ridge_gap_stays_bounded() {
    let fx = load();
    let g = fit_gamrs(&fx);
    let m = &fx.mgcv_output.gam_reml;
    report(&fx, &g);

    assert!(g.converged, "scat outer reported non-convergence");

    // ν is NOT the mechanism: both fitters land on the same degrees of freedom.
    // Observed 6.5557 vs mgcv 6.5487 — 0.1%.
    let nu_rel = (g.nu - m.nu).abs() / m.nu;
    assert!(
        nu_rel < 1.0e-2,
        "nu should agree with mgcv to 1%: gamrs {:.4} vs mgcv {:.4} ({:.2e})",
        g.nu,
        m.nu,
        nu_rel
    );

    // The level agrees; it is the curvature that does not. Observed 6.5e-4.
    assert!(
        g.max_rel < 1.5e-3,
        "fitted level drifted: max_rel {:.3e}",
        g.max_rel
    );

    // The open gap. gamrs stops on the descending side of the ridge, so its edf
    // sits ABOVE mgcv's straight-line limit. Observed 2.1664 vs 2.0102.
    let edf_gap = g.edf_total - m.edf;
    assert!(
        (0.05..0.30).contains(&edf_gap),
        "edf gap vs mgcv moved out of the pinned band [0.05, 0.30]: {edf_gap:+.4} \
         (gamrs {:.4}, mgcv {:.4}). Below the band means the outer optimiser now \
         walks the ridge — delete this test and un-ignore \
         `scat_flat_lambda_ridge_reaches_mgcv_optimum`. Above means it regressed.",
        g.edf_total,
        m.edf
    );
}

/// The target state, for when `src/outer.rs`'s step-halving fallback stops
/// declaring convergence at a gradient four orders of magnitude above the
/// loop's own tolerance (`outer.rs:589-598`, case (a): `grad_norm < 1e-3 ·
/// (|v|+1)` against the top-of-loop `5e-7 · (|v|+1)`).
#[test]
#[ignore = "open: gamrs stops short of the λ→∞ limit; see docs/scat_adjuster_parity_2026-08.md"]
fn scat_flat_lambda_ridge_reaches_mgcv_optimum() {
    let fx = load();
    let g = fit_gamrs(&fx);
    let m = &fx.mgcv_output.gam_reml;
    report(&fx, &g);
    assert!(
        (g.edf_total - m.edf).abs() < 0.02,
        "edf {:.4} vs mgcv {:.4}",
        g.edf_total,
        m.edf
    );
    assert!(
        g.max_abs_dollars < 50.0,
        "curve gap ${:.1} exceeds $50",
        g.max_abs_dollars
    );
}
