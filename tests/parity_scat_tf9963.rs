//! Parity lock for `scat` on the geometry that exposed the
//! `score_rank_adjustment` defect fixed in 0.13.0 (TF-9963): a **saturated**
//! CR basis — 5 distinct x against k = 5 — on a dollar-scale response with
//! heavy-tailed noise and very lopsided level counts, [6, 161, 432, 19, 2].
//!
//! Why this geometry: every `price_per_unit` the adjustments API reports is a
//! secant on a single-term `scat` fit made on the partial residuals of the
//! joint gaussian model, so scat sits directly under the dollar figures. With
//! the basis saturated the smooth can interpolate the five level means exactly,
//! so the REML minimum in λ is shallow — mgcv's own fixed-sp sweep (the
//! fixture's `sp_ladder`) spends only 0.04 REML units over the three decades
//! above the optimum. That makes the term a sensitive detector for anything
//! that mis-scales the penalty: with `TDist::score_rank_adjustment` returning
//! −1 the real term converged to edf 4.02 at ~30× less penalty and was 3.3%
//! off in dollars, while the three synthetic fixtures in `parity_scat.rs`
//! passed at 5e-2 throughout.
//!
//! Unlike `parity_scat_flat_ridge.rs`, this fixture HAS an interior λ optimum
//! — `saturated_basis_fixture_has_an_interior_lambda_optimum` asserts it. The
//! two fixtures catch different failure modes: a *direction/scale-of-penalty*
//! defect here, an *early-stop* defect there.
//!
//! The fixture carries mgcv's answer from all three arms mgcv can produce here
//! (`gam`+REML, `bam`+REML, `bam`+fREML). They agree with each other to 1.2e-4
//! on the fitted level, which is the point: on this term mgcv's estimator word
//! does not matter, so a single bound covers all three.
//!
//! Fixture: fully synthetic — `scripts/r/gen_scat_saturated_basis_fixture.R`,
//! seed 165. No customer data (real housing residuals stay in the gitignored
//! `data/` tree — same constraint as `docs/scat_parity_bug.md`).

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    unique_x: Vec<f64>,
    mgcv_output: MgcvArms,
    sp_ladder: Vec<Rung>,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<f64>,
    y_train: Vec<f64>,
    k: usize,
}

#[derive(Deserialize)]
struct MgcvArms {
    gam_reml: Arm,
    bam_reml: Arm,
    bam_freml: Arm,
}

#[derive(Deserialize)]
struct Arm {
    sp: f64,
    edf: f64,
    nu: f64,
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
    p.push("tests/fixtures/1d_scat_saturated_basis_n620_k5_cr.json");
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn max_rel(pred: &[f64], target: &[f64]) -> f64 {
    assert_eq!(pred.len(), target.len());
    pred.iter()
        .zip(target)
        .map(|(a, b)| (a - b).abs() / b.abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn tf9963_garage_spaces_scat_lands_on_mgcv_optimum() {
    let fx = load();
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 1), fx.inputs.x_train.clone()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let y_var = {
        let mean = y.iter().sum::<f64>() / (n as f64);
        y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n as f64)
    };

    let fit = gamrs::fit(
        gamrs::family::tdist_identity(5.0, y_var * 0.1),
        x.view(),
        y.view(),
        None,
        fx.inputs.k,
    )
    .expect("gamrs::fit (scat) should not fail");
    assert!(fit.converged, "scat outer did not converge");

    let ux = Array2::from_shape_vec((fx.unique_x.len(), 1), fx.unique_x.clone()).unwrap();
    let pred = fit.predict(ux.view()).expect("predict should not fail");
    let pred = pred.as_slice().unwrap();

    let arms: [(&str, &Arm); 3] = [
        ("gam/REML", &fx.mgcv_output.gam_reml),
        ("bam/REML", &fx.mgcv_output.bam_reml),
        ("bam/fREML", &fx.mgcv_output.bam_freml),
    ];
    println!(
        "[scat saturated basis] gamrs edf = {:.4}; σ̂ = {:.1}; iters = {}",
        fit.edf_total,
        fit.scale.sqrt(),
        fit.n_iters
    );
    for (tag, arm) in arms {
        println!(
            "  vs mgcv {tag:9} (sp {:.4e}, edf {:.4}, ν {:.4}): max_rel = {:.3e}",
            arm.sp,
            arm.edf,
            arm.nu,
            max_rel(pred, &arm.predictions_unique_x)
        );
    }

    // Bar 1e-3 against each mgcv arm: observed 5.1e-4 / 3.9e-4 / 3.8e-4 —
    // closest to bam+fREML, which is the arm the engine actually runs. With
    // `score_rank_adjustment` returning −1 this fixture sits at 1.4e-2, so the
    // bound is an order of magnitude inside where the defect lived.
    for (tag, arm) in arms {
        let rel = max_rel(pred, &arm.predictions_unique_x);
        assert!(
            rel < 1e-3,
            "scat vs mgcv {tag}: max_rel {rel:.3e} exceeds 1e-3"
        );
    }

    // edf is the diagnostic that actually names the failure mode: the defect
    // buys spurious degrees of freedom with too little penalty (edf 3.39 here,
    // 4.02 on the real term it was found on). Bracket rather than pin a
    // decimal — mgcv's own three arms span 2.3826-2.3879, gamrs sits at 2.3632.
    assert!(
        (2.2..2.6).contains(&fit.edf_total),
        "scat edf {:.4} outside mgcv's 2.38 neighbourhood — \
         the log|λS| rank convention is the first thing to check",
        fit.edf_total
    );
}

/// The fixture really has an INTERIOR λ optimum — mgcv's own score rises on
/// both sides of the sp it reports. That is what separates this fixture from
/// `parity_scat_flat_ridge.rs` (whose score descends monotonically to the
/// λ→∞ limit); if it ever stops holding, the two are testing the same regime
/// and the edf bracket above means nothing.
#[test]
fn saturated_basis_fixture_has_an_interior_lambda_optimum() {
    let fx = load();
    let at_opt = &fx.mgcv_output.gam_reml;
    let best = fx
        .sp_ladder
        .iter()
        .min_by(|a, b| a.reml.total_cmp(&b.reml))
        .expect("ladder must be non-empty");
    for r in &fx.sp_ladder {
        println!(
            "  sp={:<12.4e} reml={:>12.4} edf={:.4}",
            r.sp, r.reml, r.edf
        );
    }
    assert!(
        (best.sp - at_opt.sp).abs() / at_opt.sp < 1e-9,
        "the ladder's best rung (sp {:.4e}) is not mgcv's reported optimum (sp {:.4e})",
        best.sp,
        at_opt.sp
    );
    let (first, last) = (
        fx.sp_ladder.first().expect("non-empty"),
        fx.sp_ladder.last().expect("non-empty"),
    );
    assert!(
        first.reml > at_opt.reml && last.reml > at_opt.reml,
        "REML should be worse at both ends of the ladder ({:.4} / {:.4}) than at \
         the optimum ({:.4}) — this fixture is meant to have an interior optimum",
        first.reml,
        last.reml,
        at_opt.reml
    );
    // …and the basis really is saturated: the over-penalised end collapses to
    // the straight line, the under-penalised end spends the whole basis.
    assert!(last.edf < 2.01, "top of ladder edf {:.4}", last.edf);
    assert!(first.edf > 4.5, "bottom of ladder edf {:.4}", first.edf);
}

/// `method="fREML"` must not be a silent no-op on the scat path.
///
/// The shape-parameter driver runs damped Newton over `[ρ, log σ², log(ν−3)]`
/// and has no Fellner-Schall branch, so a fREML request used to be dropped and
/// REML/fREML came out bit-identical. The Rust API now says so; the Python
/// wrapper catches this same error, warns, and refits on REML, so callers keep
/// working while the running optimiser stops diverging from the declared one.
/// This lives beside the parity lock because scat is the family that defaulted
/// to `"fREML"` in the Python wrapper, which made every scat fit ever made
/// declare an optimiser it never ran.
#[test]
fn tf9963_scat_refuses_fellner_schall_instead_of_ignoring_it() {
    let fx = load();
    let n = fx.inputs.y_train.len();
    let x = Array2::from_shape_vec((n, 1), fx.inputs.x_train.clone()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());

    gamrs::outer::set_algorithm_override(gamrs::outer::OuterAlgorithm::FellnerSchall);
    let res = gamrs::fit(
        gamrs::family::tdist_identity(5.0, 1.0e9),
        x.view(),
        y.view(),
        None,
        fx.inputs.k,
    );
    gamrs::outer::clear_algorithm_override();

    let msg = match res {
        Err(e) => e.to_string(),
        Ok(_) => panic!("scat + Fellner-Schall must be refused, not silently ignored"),
    };
    assert!(
        msg.contains("fREML") && msg.contains("shape-parameter"),
        "error should name the parameter and the path; got {msg:?}"
    );
}
