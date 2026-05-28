//! Wall-time benchmark: gamrs `fit(gaussian_identity(), …)` re-runs on
//! the same fixture. Just measures gamrs's own throughput; the v0.x
//! comparison runs via Python (`scripts/python/bench_gamrs_vs_v0x.py`).
//!
//! `cargo bench -p gamrs --bench bench_gaussian` (release profile).

use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array1, Array2};
use serde_json::Value;

fn main() {
    let fx_name = "1d_gaussian_smooth_n2000_k30_cr";
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../tests/parity/fixtures");
    p.push(format!("{fx_name}.json"));
    let txt = std::fs::read_to_string(&p).expect("missing fixture");
    let v: Value = serde_json::from_str(&txt).unwrap();
    let xs: Vec<f64> = v["inputs"]["x_train"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap()[0].as_f64().unwrap())
        .collect();
    let ys: Vec<f64> = v["inputs"]["y_train"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let k = v["inputs"]["k"][0].as_u64().unwrap() as usize;
    let n = xs.len();
    // Current fit API expects (n, n_features) — pack the 1-D column.
    let x = Array2::from_shape_vec((n, 1), xs).unwrap();
    let y = Array1::from_vec(ys);

    // Warm up.
    let _ = gamrs::fit(
        gamrs::family::gaussian_identity(),
        x.view(),
        y.view(),
        None,
        k,
    )
    .unwrap();

    let runs = 200;
    let t0 = Instant::now();
    for _ in 0..runs {
        let _ = gamrs::fit(
            gamrs::family::gaussian_identity(),
            x.view(),
            y.view(),
            None,
            k,
        )
        .unwrap();
    }
    let elapsed = t0.elapsed();
    let per_fit = elapsed / runs;
    println!(
        "fixture: {fx_name}; n={}; k={}; gamrs per-fit = {:?}",
        x.nrows(),
        k,
        per_fit
    );
}
