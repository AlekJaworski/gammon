//! Wall-time benchmark: NegBin profile-θ fits (1-D + 2-D additive).
//! Used to track the outer-Newton iter-count and per-iter cost vs
//! mgcv_rust's NegBin path (`src/smooth.rs:3562-3637`).

use std::path::PathBuf;
use std::time::Instant;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::negbin_log;
use gamrs::fit_with_design;

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

fn load(name: &str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
}

fn bench_1d() {
    let fx = load("1d_nb_log_n300_k10_cr");
    let x_vec: Vec<f64> = fx.inputs.x_train.iter().map(|r| r[0]).collect();
    let n = x_vec.len();
    let x = Array2::from_shape_vec((n, 1), x_vec).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let k = fx.inputs.k[0];

    // Warm up
    for _ in 0..3 {
        let _ = gamrs::fit(gamrs::family::negbin_log(5.0), x.view(), y.view(), None, k).unwrap();
    }
    let mut times: Vec<f64> = Vec::new();
    let mut iters_v = 0;
    for _ in 0..30 {
        let t = Instant::now();
        let fit = gamrs::fit(gamrs::family::negbin_log(5.0), x.view(), y.view(), None, k).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        iters_v = fit.n_iters;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    println!(
        "[1D NB n=300 k=10] median={:.2}ms min={:.2}ms iters={}",
        median, min, iters_v
    );
}

fn bench_2d() {
    let fx = load("2d_nb_log_n600_k8_cr");
    let n = fx.inputs.x_train.len();
    let d = fx.inputs.x_train[0].len();
    let x_flat: Vec<f64> = fx
        .inputs
        .x_train
        .iter()
        .flat_map(|r| r.iter().copied())
        .collect();
    let x = Array2::from_shape_vec((n, d), x_flat).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train.clone());
    let ks: Vec<usize> = fx.inputs.k.clone();

    let terms = vec![
        TermSpec::Cr { col: 0, k: ks[0] },
        TermSpec::Cr { col: 1, k: ks[1] },
    ];

    // Warm
    for _ in 0..3 {
        let _ = fit_with_design(
            negbin_log(5.0),
            Additive {
                terms: terms.clone(),
            },
            x.view(),
            y.view(),
            None,
        )
        .unwrap();
    }
    let mut times: Vec<f64> = Vec::new();
    let mut iters_v = 0;
    for _ in 0..20 {
        let t = Instant::now();
        let fit = fit_with_design(
            negbin_log(5.0),
            Additive {
                terms: terms.clone(),
            },
            x.view(),
            y.view(),
            None,
        )
        .unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        iters_v = fit.n_iters;
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    println!(
        "[2D NB n=600 k=8x2] median={:.2}ms min={:.2}ms iters={}",
        median, min, iters_v
    );
}

fn main() {
    println!("[bench_nb] NegBin profile-θ benchmarks");
    bench_1d();
    bench_2d();
}
