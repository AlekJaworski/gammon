//! Scat profiling driver — runs the canonical scat 1d n=2000 k=10 fit
//! many times for use with `samply` / `cargo flamegraph`.
//!
//! Usage:
//!   cargo build --profile profiling --bench bench_scat_profile
//!   samply record target/profiling/deps/bench_scat_profile-<hash>
//!   samply load profile.json   (opens in profiler.firefox.com)

use ndarray::{Array1, Array2};

use gamrs::family::tdist_identity;

fn synth_scat(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    // Box-Muller normal — matches Python's `RNG.normal(0, 0.3, n)` more
    // closely than the Cauchy clip-tail. Scat fits Gaussian-like data
    // ~5× faster than heavy-tailed, so use Normal·0.3 here just for
    // profiling — the underlying ops we're profiling are the same.
    let mut x_flat = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = next() * 10.0;
        x_flat.push(x);
        let eta = x.sin();
        let u1 = next().max(1e-10);
        let u2 = next();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        ys.push(eta + 0.3 * z);
    }
    (
        Array2::from_shape_vec((n, 1), x_flat).unwrap(),
        Array1::from_vec(ys),
    )
}

fn main() {
    let n = 2000;
    let (x, y) = synth_scat(n, 0xdead_beef_2026_0603);

    // Warm-up.
    for _ in 0..3 {
        let _ = gamrs::fit(tdist_identity(5.0, 1.0), x.view(), y.view(), None, 10);
    }

    // Profiling loop — at ~27 ms / fit, 200 iters ≈ 5.4 s of profiled work.
    let n_runs = 200;
    let t0 = std::time::Instant::now();
    let mut total_iters = 0usize;
    for _ in 0..n_runs {
        let fit = gamrs::fit(tdist_identity(5.0, 1.0), x.view(), y.view(), None, 10)
            .expect("scat fit must converge");
        total_iters += fit.n_iters;
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "{n_runs} fits in {:.2?} ({:.1} ms/fit; {:.1} outer iters / fit)",
        elapsed,
        elapsed.as_secs_f64() * 1000.0 / (n_runs as f64),
        (total_iters as f64) / (n_runs as f64),
    );
    eprintln!();
    eprintln!("phase breakdown:");
    let _ = gamrs::profile::dump(&mut std::io::stderr());
}
