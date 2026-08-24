//! Real-data parity lock for `scat` on the term that exposed the
//! `score_rank_adjustment` defect: `garage_spaces` from the TF-9963
//! adjustments population (see the fixture's own `provenance` field).
//!
//! Why this term and not another: every `price_per_unit` the adjustments API
//! reports is a secant on a single-term `scat` fit made on the partial
//! residuals of the joint gaussian model, so scat sits directly under the
//! dollar figures. `garage_spaces` has 5 distinct x with counts
//! [6, 161, 432, 19, 2] against k=5, so the basis is saturated and the REML
//! ridge in λ is shallow — mgcv's own fixed-sp sweep moves only 0.54 REML
//! units across two decades of sp. That makes the term a sensitive detector:
//! with `TDist::score_rank_adjustment` returning −1 gamrs converged to
//! edf 4.02 (mgcv's own answer at ~30× less penalty) and was 3.3% off in
//! dollars; with the trait default it converges to edf ≈ 2.39 and is 0.03%
//! off. A synthetic fixture would not have caught it — the three
//! `parity_scat.rs` fixtures all passed at 5e-2 throughout.
//!
//! The fixture carries mgcv's answer from all three arms mgcv can produce
//! here (`gam`+REML, `bam`+REML, `bam`+fREML). They agree with each other to
//! within 0.04%, which is the point: on this term mgcv's estimator word does
//! not matter, so a single bound covers all three.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    unique_x: Vec<f64>,
    mgcv_output: MgcvArms,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
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
    predictions_unique_x: Vec<f64>,
}

fn load() -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/tf9963_garage_spaces_scat.json");
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
    let x = Array2::from_shape_vec(
        (n, 1),
        fx.inputs.x_train.iter().map(|r| r[0]).collect::<Vec<_>>(),
    )
    .unwrap();
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
        fx.inputs.k[0],
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
        "[tf9963 garage_spaces] gamrs edf = {:.4}; σ̂ = {:.1}; iters = {}",
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

    // Bar 1e-3 against each mgcv arm: observed 5.8e-4 / 3.7e-4 / 2.0e-4 —
    // closest to bam+fREML, which is the arm the engine actually runs.
    // Pre-fix this term sat at 3.3e-2, so the bound is 30× inside where the
    // defect lived.
    for (tag, arm) in arms {
        let rel = max_rel(pred, &arm.predictions_unique_x);
        assert!(
            rel < 1e-3,
            "scat vs mgcv {tag}: max_rel {rel:.3e} exceeds 1e-3"
        );
    }

    // edf is the diagnostic that actually names the failure mode: the defect
    // showed as 4.02 against mgcv's 2.37-2.43, i.e. two spurious degrees of
    // freedom bought with ~30× too little penalty. Bracket it rather than
    // pinning a decimal — mgcv's own three arms span 2.4107-2.4299.
    assert!(
        (2.2..2.6).contains(&fit.edf_total),
        "scat edf {:.4} outside mgcv's 2.41-2.43 neighbourhood — \
         the log|λS| rank convention is the first thing to check",
        fit.edf_total
    );
}
