//! Gamrs wall-clock baseline across the solver-path spectrum.
//!
//! Six fixtures, one per representative solver path:
//!
//! | Fixture                              | Path                                                     |
//! |--------------------------------------|----------------------------------------------------------|
//! | 1d_gaussian_smooth_n1000_k50_cr      | GaussianClosedFormInner (one Cholesky per outer evaluation) |
//! | 1d_bernoulli_logit_n1000_k10_cr      | PirlsInner (canonical)                                   |
//! | 1d_poisson_log_n300_k10_cr           | PirlsInner (different family, same path)                 |
//! | 1d_gamma_log_n300_k10_cr             | PirlsInner + profile-σ² Newton                           |
//! | 1d_tweedie_log_n300_k10_cr           | ShapeAwareEnvelopeScore + OwnedByLossProfile             |
//! | 1d_invgauss_log_n300_k10_cr          | PirlsInner + Newton-W (Tk·KK')                           |
//!
//! Plus a basis-evaluation micro-bench so we can attribute time to
//! "design construction" vs. "outer/inner optimisation". All
//! measurements use `std::time::Instant` with warmup + N samples per
//! fixture; the existing `bench_gaussian` follows the same pattern
//! because `criterion` is not in gamrs's dev-dependencies.
//!
//! `cargo bench -p gamrs --bench bench_baseline` (release profile).
//!
//! The bench writes timings to stdout and saves a JSON summary to
//! `/tmp/gamrs_bench_baseline.json` so the parent agent can post-process.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use serde_json::Value;

const WARMUP_ITERS: usize = 3;

/// Per-fixture sample count. Smaller for the heavy IG/Tweedie paths so
/// the whole bench finishes in under a minute on a typical workstation.
fn sample_count(name: &str) -> usize {
    if name.contains("gaussian_smooth_n1000_k50") {
        50
    } else if name.contains("invgauss") || name.contains("tweedie") {
        30
    } else {
        50
    }
}

struct Fixture {
    name: &'static str,
    /// 1-D column view of x — used by the design-prep micro-bench which
    /// calls `CrSpline::with_quantile_knots` directly.
    x1: Array1<f64>,
    /// 2-D `(n, 1)` matrix — what the canonical `gamrs::fit` API expects.
    x2: Array2<f64>,
    y: Array1<f64>,
    k: usize,
}

fn load_fixture(name: &'static str) -> Fixture {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../tests/parity/fixtures");
    p.push(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("missing fixture {name}: {e}"));
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
    let x1 = Array1::from_vec(xs.clone());
    let x2 = Array2::from_shape_vec((n, 1), xs).unwrap();
    Fixture {
        name,
        x1,
        x2,
        y: Array1::from_vec(ys),
        k,
    }
}

/// Sample N runs and return (median, p10, p90, mean) durations.
fn sample<F: FnMut()>(mut f: F, n: usize) -> (Duration, Duration, Duration, Duration) {
    let mut samples: Vec<Duration> = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }
    samples.sort();
    let median = samples[n / 2];
    let p10 = samples[(n as f64 * 0.10) as usize];
    let p90 = samples[((n as f64 * 0.90) as usize).min(n - 1)];
    let sum: Duration = samples.iter().sum();
    let mean = sum / (n as u32);
    (median, p10, p90, mean)
}

/// Result of one fixture's bench run.
struct BenchResult {
    fixture: &'static str,
    n: usize,
    k: usize,
    path: &'static str,
    median_us: f64,
    p10_us: f64,
    p90_us: f64,
    mean_us: f64,
    n_iters: usize,
    converged: bool,
    samples: usize,
}

impl BenchResult {
    fn json(&self) -> String {
        format!(
            "  {{\"fixture\":\"{}\",\"n\":{},\"k\":{},\"path\":\"{}\",\
             \"median_us\":{:.1},\"p10_us\":{:.1},\"p90_us\":{:.1},\
             \"mean_us\":{:.1},\"n_iters\":{},\"converged\":{},\"samples\":{}}}",
            self.fixture,
            self.n,
            self.k,
            self.path,
            self.median_us,
            self.p10_us,
            self.p90_us,
            self.mean_us,
            self.n_iters,
            self.converged,
            self.samples,
        )
    }
}

fn bench_one(
    fx: &Fixture,
    path: &'static str,
    mut runner: impl FnMut(&Fixture) -> (usize, bool),
) -> BenchResult {
    // Warm up.
    let mut last_iters = 0usize;
    let mut last_conv = false;
    for _ in 0..WARMUP_ITERS {
        let (it, conv) = runner(fx);
        last_iters = it;
        last_conv = conv;
    }
    let samples = sample_count(fx.name);
    let (median, p10, p90, mean) = sample(
        || {
            std::hint::black_box(runner(fx));
        },
        samples,
    );
    BenchResult {
        fixture: fx.name,
        n: fx.x2.nrows(),
        k: fx.k,
        path,
        median_us: median.as_secs_f64() * 1e6,
        p10_us: p10.as_secs_f64() * 1e6,
        p90_us: p90.as_secs_f64() * 1e6,
        mean_us: mean.as_secs_f64() * 1e6,
        n_iters: last_iters,
        converged: last_conv,
        samples,
    }
}

fn print_table(results: &[BenchResult]) {
    println!();
    println!(
        "{:<40} {:>6} {:>4} {:<32} {:>10} {:>10} {:>10} {:>6} {:>10}",
        "fixture", "n", "k", "path", "median_ms", "p10_ms", "p90_ms", "iters", "us_per_iter",
    );
    println!("{:-<140}", "");
    for r in results {
        let us_per_iter = if r.n_iters > 0 {
            r.median_us / r.n_iters as f64
        } else {
            f64::NAN
        };
        println!(
            "{:<40} {:>6} {:>4} {:<32} {:>10.3} {:>10.3} {:>10.3} {:>6} {:>10.1}",
            r.fixture,
            r.n,
            r.k,
            r.path,
            r.median_us / 1e3,
            r.p10_us / 1e3,
            r.p90_us / 1e3,
            r.n_iters,
            us_per_iter,
        );
    }
}

// ---------------------------------------------------------------------------
// Sub-phase micro-bench: how much of the slowest fit is just basis prep
// vs. the outer solver loop?
//
// We can measure "basis evaluation + sum-to-zero + design matrix build"
// in isolation by calling `CrSpline::with_quantile_knots` + `Basis::evaluate`
// publicly. Everything else (inner solves, score grad, outer Newton)
// lives in pub(crate) helpers and is only timed via full-fit subtraction.
// ---------------------------------------------------------------------------

fn bench_design_prep(fx: &Fixture) -> Duration {
    use gamrs::basis::CrSpline;
    // Warm up.
    for _ in 0..WARMUP_ITERS {
        let cr = CrSpline::with_quantile_knots(fx.x1.view(), fx.k).unwrap();
        std::hint::black_box(cr);
    }
    let n = 200;
    let t0 = Instant::now();
    for _ in 0..n {
        let cr = CrSpline::with_quantile_knots(fx.x1.view(), fx.k).unwrap();
        std::hint::black_box(cr);
    }
    t0.elapsed() / (n as u32)
}

fn main() {
    println!("=== gamrs baseline bench (release profile) ===");
    println!(
        "rustc: {}; arch: {}; opt-level: 3 (release)",
        option_env!("RUSTC_VERSION").unwrap_or("?"),
        std::env::consts::ARCH,
    );

    let mut results: Vec<BenchResult> = Vec::new();

    // 1. Gaussian closed-form, largest n + k.
    {
        let fx = load_fixture("1d_gaussian_smooth_n1000_k50_cr");
        let r = bench_one(&fx, "GaussianClosedFormInner", |fx| {
            let f = gamrs::fit(
                gamrs::family::gaussian_identity(),
                fx.x2.view(),
                fx.y.view(),
                None,
                fx.k,
            )
            .unwrap();
            (f.n_iters, f.converged)
        });
        results.push(r);
    }

    // 2. Bernoulli logit canonical PIRLS.
    {
        let fx = load_fixture("1d_bernoulli_logit_n1000_k10_cr");
        let r = bench_one(&fx, "PirlsInner (Bernoulli)", |fx| {
            let f = gamrs::fit(
                gamrs::family::bernoulli_logit(),
                fx.x2.view(),
                fx.y.view(),
                None,
                fx.k,
            )
            .unwrap();
            (f.n_iters, f.converged)
        });
        results.push(r);
    }

    // 3. Poisson log canonical PIRLS.
    {
        let fx = load_fixture("1d_poisson_log_n300_k10_cr");
        let r = bench_one(&fx, "PirlsInner (Poisson)", |fx| {
            let f = gamrs::fit(
                gamrs::family::poisson_log(),
                fx.x2.view(),
                fx.y.view(),
                None,
                fx.k,
            )
            .unwrap();
            (f.n_iters, f.converged)
        });
        results.push(r);
    }

    // 4. Gamma log + profile-sigma2 Newton.
    {
        let fx = load_fixture("1d_gamma_log_n300_k10_cr");
        let r = bench_one(&fx, "PirlsInner + profile-σ² Newton", |fx| {
            let f = gamrs::fit(
                gamrs::family::gamma_log(),
                fx.x2.view(),
                fx.y.view(),
                None,
                fx.k,
            )
            .unwrap();
            (f.n_iters, f.converged)
        });
        results.push(r);
    }

    // 5. Tweedie: ShapeAwareEnvelopeScore + OwnedByLossProfile.
    {
        let fx = load_fixture("1d_tweedie_log_n300_k10_cr");
        let r = bench_one(&fx, "ShapeAwareEnvelopeScore (Tweedie)", |fx| {
            let f = gamrs::fit(
                gamrs::family::tweedie_log(1.5, 1.0),
                fx.x2.view(),
                fx.y.view(),
                None,
                fx.k,
            )
            .unwrap();
            (f.n_iters, f.converged)
        });
        results.push(r);
    }

    // 6. Inverse Gaussian: PIRLS + Newton-W (Tk·KK') — known-expensive Newton path.
    {
        let fx = load_fixture("1d_invgauss_log_n300_k10_cr");
        let r = bench_one(&fx, "PirlsInner + Newton-W (IG)", |fx| {
            let f = gamrs::fit(
                gamrs::family::inverse_gaussian_log(),
                fx.x2.view(),
                fx.y.view(),
                None,
                fx.k,
            )
            .unwrap();
            (f.n_iters, f.converged)
        });
        results.push(r);
    }

    print_table(&results);

    // Sub-phase: design-prep timings for each fixture.
    println!();
    println!("=== sub-phase: CR-spline quantile-knot design prep (per call) ===");
    println!(
        "{:<40} {:>12} {:>14}",
        "fixture", "design_us", "% of full fit",
    );
    println!("{:-<70}", "");
    let fixtures_for_subphase = [
        "1d_gaussian_smooth_n1000_k50_cr",
        "1d_bernoulli_logit_n1000_k10_cr",
        "1d_poisson_log_n300_k10_cr",
        "1d_gamma_log_n300_k10_cr",
        "1d_tweedie_log_n300_k10_cr",
        "1d_invgauss_log_n300_k10_cr",
    ];
    let mut design_us_by_name: Vec<(String, f64)> = Vec::new();
    for name in fixtures_for_subphase {
        let fx = load_fixture(name);
        let dur = bench_design_prep(&fx);
        let us = dur.as_secs_f64() * 1e6;
        // Look up the full-fit median.
        let full_us = results
            .iter()
            .find(|r| r.fixture == name)
            .map(|r| r.median_us)
            .unwrap_or(f64::NAN);
        let pct = 100.0 * us / full_us;
        println!("{:<40} {:>12.2} {:>13.2}%", name, us, pct);
        design_us_by_name.push((name.to_string(), us));
    }

    // Identify the slowest fit so the parent agent knows where to focus.
    let slowest = results
        .iter()
        .max_by(|a, b| a.median_us.partial_cmp(&b.median_us).unwrap())
        .unwrap();
    println!();
    println!(
        "[slowest fit] {} ({}, n={}, k={}) — median {:.3} ms, {} outer iters → {:.2} us/iter",
        slowest.fixture,
        slowest.path,
        slowest.n,
        slowest.k,
        slowest.median_us / 1e3,
        slowest.n_iters,
        if slowest.n_iters > 0 {
            slowest.median_us / slowest.n_iters as f64
        } else {
            f64::NAN
        },
    );

    // ----------------------------------------------------------------
    // Tweedie-specific hot-function profile (gamrs::special::*).
    //
    // No perf/flamegraph available in this worktree — instead we
    // micro-bench the Tweedie special functions that we know are called
    // every outer Newton iteration:
    //   - `tweedie_series(y, φ, p)` — full Dunn-Smyth series + deriv-
    //     w.r.t-ρ and w.r.t-p, n observations at a time, ONCE per
    //     outer iter (analytic_shape_score_gradient).
    //   - `tweedie_log_w(y, φ, p)` — Dunn-Smyth log W scalar, called
    //     once per (i, outer-iter) in `saturated_log_lik`. So total
    //     calls per fit ≈ n · n_outer_iters · (1 + halving probes).
    // The bench charges actual end-to-end cost; the call counts here
    // are nominal lower bounds, so the "% of full fit" attribution
    // below is a lower bound on the share these helpers consume.
    // ----------------------------------------------------------------
    println!();
    println!("=== Tweedie hot-function micro-bench (slowest path) ===");
    {
        let fx = load_fixture("1d_tweedie_log_n300_k10_cr");
        let y_slice: Vec<f64> = fx.y.iter().copied().collect();
        let phi = 0.975; // post-fit value from parity test diagnostics
        let p = 1.7; // mid-range Tweedie shape

        // tweedie_series — full vectorised call, n=300.
        for _ in 0..WARMUP_ITERS {
            let _ = gamrs::special::tweedie_series(&y_slice, phi, p);
        }
        let n_series = 200;
        let t0 = Instant::now();
        for _ in 0..n_series {
            let r = gamrs::special::tweedie_series(&y_slice, phi, p);
            std::hint::black_box(r);
        }
        let series_us = t0.elapsed().as_secs_f64() * 1e6 / n_series as f64;

        // tweedie_log_w — scalar; bench the inner per-obs cost summed across n.
        for _ in 0..WARMUP_ITERS {
            let mut s = 0.0;
            for &yi in &y_slice {
                s += gamrs::special::tweedie_log_w(yi, phi, p);
            }
            std::hint::black_box(s);
        }
        let n_logw = 200;
        let t0 = Instant::now();
        for _ in 0..n_logw {
            let mut s = 0.0;
            for &yi in &y_slice {
                s += gamrs::special::tweedie_log_w(yi, phi, p);
            }
            std::hint::black_box(s);
        }
        let logw_sum_us = t0.elapsed().as_secs_f64() * 1e6 / n_logw as f64;

        // log_gamma / digamma — called O(j_max · n) inside tweedie_series.
        for _ in 0..WARMUP_ITERS {
            let mut s = 0.0;
            for j in 1..=200_u64 {
                s += gamrs::special::log_gamma(j as f64);
            }
            std::hint::black_box(s);
        }
        let n_lg = 5000;
        let t0 = Instant::now();
        for _ in 0..n_lg {
            let mut s = 0.0;
            for j in 1..=200_u64 {
                s += gamrs::special::log_gamma(j as f64);
            }
            std::hint::black_box(s);
        }
        let lg_us = t0.elapsed().as_secs_f64() * 1e6 / n_lg as f64;

        let tweedie_full_us = results
            .iter()
            .find(|r| r.fixture == "1d_tweedie_log_n300_k10_cr")
            .map(|r| r.median_us)
            .unwrap_or(f64::NAN);
        let n_outer = slowest.n_iters as f64;
        // Nominal call count per fit (no halving probes counted).
        let est_series_calls = n_outer; // 1 per outer iter
        let est_logw_calls_total_n_loops = n_outer; // n scalar log_w sums per outer iter

        println!(
            "{:<46} {:>10} {:>14}",
            "function", "per_call_us", "est % of fit",
        );
        println!("{:-<78}", "");
        println!(
            "{:<46} {:>10.2} {:>13.2}%",
            "tweedie_series(y[n=300], φ, p)",
            series_us,
            100.0 * series_us * est_series_calls / tweedie_full_us,
        );
        println!(
            "{:<46} {:>10.2} {:>13.2}%",
            "Σ tweedie_log_w(y[n=300], φ, p)",
            logw_sum_us,
            100.0 * logw_sum_us * est_logw_calls_total_n_loops / tweedie_full_us,
        );
        println!(
            "{:<46} {:>10.2} {:>13}",
            "log_gamma × 200 (single sweep cost)", lg_us, "—",
        );
        println!(
            "  (Tweedie full-fit median: {:.2} ms; outer iters: {})",
            tweedie_full_us / 1e3,
            slowest.n_iters,
        );
    }

    // Write JSON summary for downstream consumption.
    let json = {
        let mut s = String::from("{\n  \"fixtures\": [\n");
        for (i, r) in results.iter().enumerate() {
            s.push_str(&r.json());
            if i + 1 < results.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ],\n  \"design_prep_us\": {\n");
        for (i, (n, us)) in design_us_by_name.iter().enumerate() {
            s.push_str(&format!("    \"{}\": {:.2}", n, us));
            if i + 1 < design_us_by_name.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  }\n}\n");
        s
    };
    let out_path = "/tmp/gamrs_bench_baseline.json";
    if let Err(e) = std::fs::write(out_path, &json) {
        eprintln!("warning: failed to write {out_path}: {e}");
    } else {
        println!();
        println!("JSON summary written to {out_path}");
    }
}
