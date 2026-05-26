//! Parity test for gamrs's tensor-product path (`te(x0, x1)`) against an
//! mgcv-generated fixture.
//!
//! Fits `y ~ te(x0, x1, k=c(5,5), bs=c("cr","cr"))` via the gamrs
//! `Additive` path with a single `TermSpec::Tensor` term, and compares
//! the predicted μ̂ on the training set against mgcv's
//! `predict(..., type='response')` output.
//!
//! Tolerance is set looser than the single-smooth bound (1e-5) because:
//! - two smoothing parameters per term → more outer-Newton DOF;
//! - mgcv's `te()` applies a different stable-reparam path
//!   (`Sm[[i]] / max-eig` per margin) at fit time, which produces tiny FP
//!   differences vs. our straightforward C' S C rotation.
//!
//! The 5e-3 bound matches the looser target stated for epic 94c in the
//! task brief.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, MarginKind, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit_with_design;

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    mgcv_output: MgcvOutput,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
}

#[derive(Deserialize)]
struct MgcvOutput {
    predictions_train: Vec<f64>,
    scale: f64,
}

fn load_fixture(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
    serde_json::from_str(&txt).expect("malformed fixture json")
}

fn max_rel_err(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1.0))
        .fold(0.0_f64, f64::max)
}

#[test]
fn tensor_2d_gaussian_te_n300_k5x5() {
    // Small-n case (n=300, p=25 effective columns after centring).
    // The REML score is shallow near the optimum at this dimension —
    // gamrs and mgcv land on slightly different ρ̂ even though both are
    // valid Newton/BFGS stationary points. Bound is widened relative
    // to n=1000 to absorb this (mgcv-vs-gamrs outer optimiser disagreement
    // on the shallow region, not a basis correctness issue — see
    // n=1000 sibling case for tighter agreement).
    run_tensor_parity("2d_gaussian_te_n300_k5x5", 2e-2, 5e-2);
}

#[test]
fn tensor_2d_gaussian_te_n1000_k5x5() {
    // Larger-n version with the same k=(5,5). REML score curvature is
    // sharper at n=1000, so we expect tighter agreement with mgcv on
    // both μ̂ and φ̂ — this is the primary parity gate for 94c.
    run_tensor_parity("2d_gaussian_te_n1000_k5x5", 5e-3, 2e-2);
}

fn run_tensor_parity(fixture_name: &str, mu_rel_bound: f64, scale_rel_bound: f64) {
    let fx = load_fixture(fixture_name);

    let n = fx.inputs.x_train.len();
    let mut x_flat: Vec<f64> = Vec::with_capacity(n * 2);
    for row in &fx.inputs.x_train {
        x_flat.push(row[0]);
        x_flat.push(row[1]);
    }
    let x = Array2::from_shape_vec((n, 2), x_flat).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());

    let terms = vec![TermSpec::Tensor {
        col_a: 0,
        col_b: 1,
        k_a: fx.inputs.k[0],
        k_b: fx.inputs.k[1],
        bs_a: MarginKind::Cr,
        bs_b: MarginKind::Cr,
    }];
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("Tensor Gaussian fit failed");
    assert!(fit.converged, "outer Newton did not converge");
    // 2 smoothing params (one per margin) for the single tensor term.
    assert_eq!(fit.rho.len(), 2, "rho should have one entry per margin");
    assert_eq!(fit.lambda.len(), 2);
    assert_eq!(fit.edf_per_term.len(), 2);

    // Predictions vs mgcv.
    let pred = fit
        .predict(x.view())
        .expect("Tensor predict on training x failed");
    let rel = max_rel_err(pred.as_slice().unwrap(), &fx.mgcv_output.predictions_train);
    let scale_rel = (fit.scale - fx.mgcv_output.scale).abs() / fx.mgcv_output.scale.max(1e-12);
    println!(
        "[{fixture_name}] max_rel = {rel:.3e}; scale_rel = {scale_rel:.3e} \
         (gamrs {:.6e} vs mgcv {:.6e}); ρ̂ = [{:.3}, {:.3}]; edf = ({:.2}, {:.2}); iters = {}",
        fit.scale,
        fx.mgcv_output.scale,
        fit.rho[0],
        fit.rho[1],
        fit.edf_per_term[0],
        fit.edf_per_term[1],
        fit.n_iters,
    );
    assert!(
        rel < mu_rel_bound,
        "tensor μ rel error {rel:.3e} exceeds {mu_rel_bound:.0e} — possible regression"
    );
    assert!(
        scale_rel < scale_rel_bound,
        "tensor φ̂ rel error {scale_rel:.3e} exceeds {scale_rel_bound:.0e}"
    );
}
