//! Fit-time persistence — round-trip a `FittedGam` through
//! `serialize` → `deserialize` and confirm predictions are
//! bit-for-bit identical to the original fit. Covers each Predictor
//! variant (Cr / Re / CrStable) on a representative family so the
//! serde derives are exercised end-to-end.

use ndarray::{Array1, Axis};

use gammon::family::{bernoulli_logit, gaussian_identity, tweedie_log};
use gammon::fit::FittedGam;
use gammon::{CrStable, Re};

fn assert_predictions_identical(orig: &FittedGam, restored: &FittedGam, x: &Array1<f64>) {
    let x2 = x.view().insert_axis(Axis(1));
    let p_orig = orig.predict(x2).expect("orig predict failed");
    let p_new = restored
        .predict(x2)
        .expect("restored predict failed");
    assert_eq!(p_orig.len(), p_new.len());
    for (i, (a, b)) in p_orig.iter().zip(p_new.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "prediction mismatch at row {i}: orig={a}, restored={b}",
        );
    }
}

fn assert_vcov_identical(orig: &FittedGam, restored: &FittedGam) {
    assert_eq!(orig.vcov.dim(), restored.vcov.dim(), "vcov shape drift");
    for ((i, j), &v) in orig.vcov.indexed_iter() {
        let r = restored.vcov[[i, j]];
        assert!(
            (v - r).abs() < 1e-15,
            "vcov mismatch at [{i},{j}]: orig={v}, restored={r}"
        );
    }
}

#[test]
fn fitted_gam_bincode_roundtrip_gaussian_cr() {
    // Default design strategy is Cr — this exercises Predictor::Cr.
    let n = 200;
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, n);
    let y: Array1<f64> = x.iter().map(|&xi| (2.0 * xi).sin()).collect();

    let fit = gammon::fit(gaussian_identity(), x.view().insert_axis(Axis(1)), y.view(), None, 10)
        .expect("gaussian fit failed");

    let bytes = fit.serialize().expect("serialize failed");
    let restored = FittedGam::deserialize(&bytes).expect("deserialize failed");

    assert_predictions_identical(&fit, &restored, &x);
    assert_vcov_identical(&fit, &restored);
    assert_eq!(fit.rho.len(), restored.rho.len());
    for (a, b) in fit.rho.iter().zip(restored.rho.iter()) {
        assert!((a - b).abs() < 1e-15);
    }
    assert!((fit.scale - restored.scale).abs() < 1e-15);
    assert_eq!(fit.converged, restored.converged);
}

#[test]
fn fitted_gam_bincode_roundtrip_bernoulli_cr() {
    let n = 300;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let eta = 3.0 * (x - 0.5);
            let p = 1.0 / (1.0 + (-eta).exp());
            let h = (i.wrapping_mul(2654435761)) as u32;
            let u = (h as f64) / (u32::MAX as f64);
            if u < p { 1.0 } else { 0.0 }
        })
        .collect();
    let x = Array1::from_vec(xs);
    let y = Array1::from_vec(ys);

    let fit = gammon::fit(bernoulli_logit(), x.view().insert_axis(Axis(1)), y.view(), None, 10)
        .expect("bernoulli fit failed");
    let bytes = fit.serialize().expect("serialize failed");
    let restored = FittedGam::deserialize(&bytes).expect("deserialize failed");

    assert_predictions_identical(&fit, &restored, &x);
    assert_vcov_identical(&fit, &restored);
    // Bernoulli σ²=1 by convention — should round-trip exactly.
    assert_eq!(restored.scale, 1.0);
}

#[test]
fn fitted_gam_bincode_roundtrip_tweedie_cr() {
    // Tweedie exercises the shape-aware joint-Newton driver.
    let n = 250;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let h = (i.wrapping_mul(2654435761)) as u32;
            let u = (h as f64) / (u32::MAX as f64);
            let mu = (2.0 * x).exp();
            if u < 0.3 { 0.0 } else { mu * (1.0 + 0.5 * (u - 0.5)) }
        })
        .collect();
    let x = Array1::from_vec(xs);
    let y = Array1::from_vec(ys);

    let fit = gammon::fit(tweedie_log(1.5, 1.0), x.view().insert_axis(Axis(1)), y.view(), None, 8)
        .expect("tweedie fit failed");
    let bytes = fit.serialize().expect("serialize failed");
    let restored = FittedGam::deserialize(&bytes).expect("deserialize failed");

    assert_predictions_identical(&fit, &restored, &x);
    assert_vcov_identical(&fit, &restored);
}

#[test]
fn fitted_gam_roundtrip_re_predictor_variant() {
    // Re design exercises Predictor::Re (one-hot levels).
    let group_ids = [0.0_f64, 1.0, 2.0, 3.0];
    let n = 80;
    let xs: Vec<f64> = (0..n).map(|i| group_ids[i % group_ids.len()]).collect();
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(i, &g)| {
            let base = g * 1.5;
            let h = (i.wrapping_mul(2654435761)) as u32;
            let u = (h as f64) / (u32::MAX as f64) - 0.5;
            base + 0.2 * u
        })
        .collect();
    let x = Array1::from_vec(xs);
    let y = Array1::from_vec(ys);

    let fit = gammon::fit_with_design(gaussian_identity(), Re, x.view().insert_axis(Axis(1)), y.view(), None)
        .expect("re fit failed");
    let bytes = fit.serialize().expect("serialize failed");
    let restored = FittedGam::deserialize(&bytes).expect("deserialize failed");

    assert_predictions_identical(&fit, &restored, &x);
    assert_vcov_identical(&fit, &restored);
}

#[test]
fn fitted_gam_roundtrip_cr_stable_predictor_variant() {
    // CrStable exercises Predictor::CrStable (rotation V is the extra
    // field beyond Cr).
    let n = 200;
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, n);
    let y: Array1<f64> = x.iter().map(|&xi| (3.0 * xi).sin()).collect();

    let fit = gammon::fit_with_design(
        gaussian_identity(),
        CrStable { k: 10 },
        x.view().insert_axis(Axis(1)),
        y.view(),
        None,
    )
    .expect("cr_stable fit failed");

    let bytes = fit.serialize().expect("serialize failed");
    let restored = FittedGam::deserialize(&bytes).expect("deserialize failed");

    assert_predictions_identical(&fit, &restored, &x);
    assert_vcov_identical(&fit, &restored);
}

#[test]
fn deserialize_rejects_bad_magic() {
    let bytes = vec![b'N', b'O', b'P', b'E', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let err = match FittedGam::deserialize(&bytes) {
        Ok(_) => panic!("deserialize must reject bad magic"),
        Err(e) => e,
    };
    assert!(format!("{err}").contains("bad magic"));
}

#[test]
fn deserialize_rejects_unsupported_version() {
    // Build a header with a known-bad version.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GAMMON");
    bytes.extend_from_slice(&u32::to_le_bytes(9999));
    bytes.extend_from_slice(&u64::to_le_bytes(0));
    let err = match FittedGam::deserialize(&bytes) {
        Ok(_) => panic!("deserialize must reject unknown version"),
        Err(e) => e,
    };
    assert!(
        format!("{err}").contains("9999"),
        "error must surface the offending version: {err}"
    );
}

#[test]
fn deserialize_rejects_truncated_body() {
    let n = 50;
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, n);
    let y: Array1<f64> = x.iter().map(|&xi| (2.0 * xi).sin()).collect();
    let fit = gammon::fit(gaussian_identity(), x.view().insert_axis(Axis(1)), y.view(), None, 6).unwrap();
    let bytes = fit.serialize().unwrap();
    let truncated = &bytes[..bytes.len() - 5];
    let err = match FittedGam::deserialize(truncated) {
        Ok(_) => panic!("truncated body must error"),
        Err(e) => e,
    };
    assert!(format!("{err}").to_lowercase().contains("truncat"));
}

#[test]
fn serialize_size_smoke_gaussian_n500() {
    // Document the wire-format size on a typical fit (n=500, k=10) so
    // we notice if the schema grows unexpectedly.
    let n = 500;
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, n);
    let y: Array1<f64> = x.iter().map(|&xi| (2.0 * xi).sin()).collect();
    let fit = gammon::fit(gaussian_identity(), x.view().insert_axis(Axis(1)), y.view(), None, 10).unwrap();
    let bytes = fit.serialize().unwrap();
    println!("serialized size (Gaussian n=500, k=10): {} bytes", bytes.len());
    // Rough guard: a single-smooth Gaussian fit should fit in a few KB.
    assert!(
        bytes.len() < 20_000,
        "serialized size {} bytes exceeds 20 KB sanity cap",
        bytes.len()
    );
    assert!(bytes.len() > 100, "serialized size suspiciously small");
}

#[test]
fn json_roundtrip_is_byte_for_byte_predictions() {
    let n = 150;
    let x: Array1<f64> = Array1::linspace(0.0, 1.0, n);
    let y: Array1<f64> = x.iter().map(|&xi| (2.0 * xi).sin()).collect();
    let fit = gammon::fit(gaussian_identity(), x.view().insert_axis(Axis(1)), y.view(), None, 8).unwrap();
    let s = fit.serialize_json().unwrap();
    let restored = FittedGam::deserialize_json(&s).unwrap();
    assert_predictions_identical(&fit, &restored, &x);
}
