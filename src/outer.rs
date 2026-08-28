//! Layer 5 — outer optimiser over θ.
//!
//! `NewtonWithHalving` implements `crate::traits::OuterSolver`. ρ-dim
//! generic via the trait surface; the concrete Phase-0 path always has
//! `dim = 1` but the same impl handles multi-θ (joint ρ + log σ²) once
//! Phase 1 wires it.
//!
//! The optimiser consumes `ScoreDerivatives::value_grad_hess` and never
//! FD-probes `value_and_grad` directly — the Hessian-FD is the score's
//! responsibility (architecture-assumptions.md §B5). This is the
//! structural defence against the closed-form-vs-FD drift bug class.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, Solve, UPLO};

use crate::error::{GamrsError, Result};
use crate::traits::{OuterFit, OuterSolver, ScoreDerivatives};

pub struct NewtonOpts {
    pub max_iters: usize,
    /// Score-relative gradient tolerance — `|grad|_∞ < grad_tol·(|REML|+1)`.
    /// Matches mgcv's `gam.fit3.r:1644-1645` `conv.tol=1e-7` × 5 default.
    pub grad_tol: f64,
    /// Score-change tolerance — relative `|ΔREML|/|REML| < reml_tol`. Active
    /// only after iter ≥ 3, mirroring mgcv's same-line condition.
    pub reml_tol: f64,
    pub step_min: f64,
    /// Lower bound on the (positive) Hessian eigenvalues. Negative or
    /// near-zero curvature is replaced with this floor so Newton's step
    /// direction is always descent.
    pub hess_floor: f64,
    /// Cap on the L_∞ norm of a single Newton step in θ-space — keeps the
    /// outer loop from leaping into the degenerate-σ² regime.
    pub max_step: f64,
}

/// Family-facing outer-Newton tuning. The fit drivers consume this via
/// `Loss::outer_tuning()` and convert to [`NewtonOpts`]. Centralises the
/// per-family overrides mgcv ships per family (gam.fit3.r `conv.tol`,
/// `max.half`, step bounds) — defaults are mgcv-parity.
///
/// To override per family, implement `Loss::outer_tuning()` and return
/// a different struct. The base `OuterTuning::mgcv_default()` matches
/// mgcv's `conv.tol=1e-7 × 5` (gradient) and `conv.tol=1e-7` (REML
/// change) — verified against mgcv_rust smooth.rs:2545-2569.
#[derive(Debug, Clone, Copy)]
pub struct OuterTuning {
    /// Score-relative gradient tolerance: converge when
    /// `|grad|_∞ < grad_tol · (|REML| + 1)`.
    ///
    /// mgcv's extended-family outer Newton uses
    /// `conv.tol = .Machine$double.eps^.5` ≈ 1.49e-8 (`fast-REML.r:1481`),
    /// which is what `scat` goes through. The 5e-7 this used to carry was
    /// sourced from mgcv_rust (`smooth.rs:2545-2569`) — the port, not mgcv —
    /// and is 34× looser. Because the bound is scaled by the SCORE and the
    /// score is O(10²-10³) while the λ ridge can be flat to 1e-5, 5e-7 made
    /// the stopping ball wide enough to hold a real curve difference: on the
    /// real saturated-basis term the optimiser halted at |g|∞ = 3.84e-4
    /// against a bound of 4.31e-4, at a point whose score was 8.6e-6 WORSE
    /// than mgcv's. Measured against mgcv on the same (standardized) problem,
    /// tightening to `sqrt(eps)` moves the worst term from 1.09e-4 to 3.81e-5.
    ///
    /// NOTE this is DELIBERATELY ~336× tighter than mgcv's own rule and is NOT
    /// justified by parity — mgcv's `newton()` uses `conv.tol = 1e-6`
    /// (`mgcv.r:2209`) and tests `abs(grad) > score.scale*conv.tol*5`
    /// (`gam.fit3.r:1644`), i.e. `5e-6·score.scale`. The earlier citation of
    /// `fast-REML.r:1481` was wrong: that is `fast.REML.fit`, the Gaussian
    /// fREML path, not the one scat takes.
    ///
    /// It is justified EMPIRICALLY, and re-measured after the `max_half` fix
    /// (which invalidated the first measurement). Against mgcv 1.9.4 run
    /// directly on the ten real adjuster terms, same standardized problem:
    ///
    /// ```text
    ///                    sqrt(eps)      mgcv's 5e-6
    ///   garage_spaces    3.476e-5       5.882e-4      ($20.6 vs $348)
    ///   condition        3.381e-5       1.016e-4      ($17 vs $51)
    ///   worst term       3.809e-5       5.882e-4
    /// ```
    ///
    /// Cost: one extra outer iteration and ~10% wall clock (51-53 ms/fit at
    /// mgcv's tolerance vs 56-59 at this one, `bench_scat_profile`). A 15×
    /// accuracy gain on the worst term is worth that; revisit if the trade
    /// changes.
    pub grad_tol: f64,
    /// Score-relative REML-change tolerance: converge when
    /// `|ΔREML| / max(|REML|, 1) < reml_tol`. Active after iter ≥ 3.
    /// mgcv default 1e-7.
    pub reml_tol: f64,
    /// Cap on outer iterations before the optimiser bails out with
    /// `NotConverged`. mgcv default 200.
    pub max_iters: usize,
    /// L∞ cap on a single Newton step in θ-space. Per-axis caps from
    /// `axis_step_caps` override this when present. mgcv default 5.0.
    pub max_step: f64,
}

impl OuterTuning {
    /// mgcv R / mgcv_rust default tolerances. Match
    /// `gam.fit3.r:1644 conv.tol=1e-7` and mgcv_rust's
    /// `smooth.rs:2545-2569 reml.tol`.
    pub fn mgcv_default() -> Self {
        Self {
            grad_tol: f64::EPSILON.sqrt(),
            reml_tol: 1.0e-7,
            max_iters: 200,
            max_step: 5.0,
        }
    }

    /// Convert to the lower-level [`NewtonOpts`] the outer Newton actually
    /// consumes. Fills in the less-commonly-tuned fields (`step_min`,
    /// `hess_floor`) with executor-side defaults.
    pub fn to_newton_opts(self) -> NewtonOpts {
        NewtonOpts {
            max_iters: self.max_iters,
            grad_tol: self.grad_tol,
            reml_tol: self.reml_tol,
            step_min: 1e-3,
            hess_floor: 1e-8,
            max_step: self.max_step,
        }
    }
}

impl Default for OuterTuning {
    fn default() -> Self {
        Self::mgcv_default()
    }
}

thread_local! {
    /// Thread-local override for the per-family [`OuterTuning`]. When
    /// `Some`, fit drivers use this in place of the family's
    /// `Loss::outer_tuning()`. Set via [`set_tuning_override`] and cleared
    /// via [`clear_tuning_override`]. Intended for the tolerance-sweep
    /// script — production callers leave it `None`.
    static OUTER_TUNING_OVERRIDE: std::cell::Cell<Option<OuterTuning>> =
        const { std::cell::Cell::new(None) };
}

/// Set the thread-local outer-tuning override. All subsequent fits on
/// this thread use the override instead of `Loss::outer_tuning()` until
/// [`clear_tuning_override`] is called.
pub fn set_tuning_override(tuning: OuterTuning) {
    OUTER_TUNING_OVERRIDE.with(|c| c.set(Some(tuning)));
}

/// Clear the thread-local outer-tuning override; subsequent fits revert
/// to `Loss::outer_tuning()`.
pub fn clear_tuning_override() {
    OUTER_TUNING_OVERRIDE.with(|c| c.set(None));
}

/// Resolve the active tuning: override if set, otherwise the family default.
/// Used by every fit driver.
pub fn resolve_tuning<L: crate::traits::Loss>(loss: &L) -> OuterTuning {
    OUTER_TUNING_OVERRIDE
        .with(|c| c.get())
        .unwrap_or_else(|| loss.outer_tuning())
}

/// Which outer optimiser to use. Maps to mgcv R's `method` argument.
///
/// - `Newton`: damped Newton on the REML score (the gamrs default — mgcv
///   `gam()` equivalent). Per-iter cost: inner PIRLS + analytic Hessian
///   assembly. Best for small-to-mid n (≤ ~10K) where the per-iter cost
///   pays off via fast convergence.
/// - `FellnerSchall`: Wood & Fasiolo (2017) multiplicative λ update
///   (mgcv `bam()`'s `fREML` equivalent). Per-iter cost: inner PIRLS + a
///   few O(p²) traces. Cheaper per-iter, slightly more iters; wins at
///   large n on GLM families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterAlgorithm {
    Newton,
    FellnerSchall,
}

thread_local! {
    static OUTER_ALGORITHM_OVERRIDE: std::cell::Cell<Option<OuterAlgorithm>> =
        const { std::cell::Cell::new(None) };
}

pub fn set_algorithm_override(algo: OuterAlgorithm) {
    OUTER_ALGORITHM_OVERRIDE.with(|c| c.set(Some(algo)));
}

pub fn clear_algorithm_override() {
    OUTER_ALGORITHM_OVERRIDE.with(|c| c.set(None));
}

pub fn resolved_algorithm() -> OuterAlgorithm {
    OUTER_ALGORITHM_OVERRIDE
        .with(|c| c.get())
        .unwrap_or(OuterAlgorithm::Newton)
}

/// The text every Fellner-Schall refusal starts with. The Python wrapper
/// matches on it to turn the refusal into a REML fallback — keep it in step
/// with `python/gamrs/_fitter.py::FS_UNAVAILABLE_MARKER`. The unit test at
/// the bottom of this module fails if the rendered message stops carrying it.
pub const FS_UNAVAILABLE_MARKER: &str =
    "method='fREML' (Fellner-Schall) is not implemented for the";

/// Say so when a driver that has no Fellner-Schall branch is asked for one,
/// instead of running Newton and letting the caller believe otherwise.
///
/// Fellner-Schall is a real, released solver (0.6.0) and is honoured wherever
/// it was ported: [`crate::fit::driver::fit_pirls_envelope`], the GLM envelope
/// driver, dispatches on [`resolved_algorithm`] and runs it. The port never
/// reached four other drivers — the gaussian closed-form path, the quantile
/// path, the profile-shape path (negbin) and the joint shape-parameter path
/// (scat, tweedie, ocat) — and those called `NewtonWithHalving` regardless, so
/// `method="fREML"` on them was a silent no-op. This is the loud version.
///
/// Direct Rust callers get the error. The Python wrapper catches it, warns,
/// and refits on REML, because a library that raises on a parameter it used
/// to accept breaks callers for no statistical gain: gamrs' `fREML` is the
/// Fellner-Schall optimiser (Wood & Fasiolo 2017) and mgcv's
/// `bam(method="fREML")` is the REML criterion computed the fast way — score
/// two sp vectors with `sp` pinned and bam's fREML and REML criteria return
/// identical numbers. Both are routes to the same criterion, and damped
/// Newton on the REML score is the stronger route of the two here.
pub fn reject_unsupported_algorithm(path: &str) -> crate::error::Result<()> {
    match resolved_algorithm() {
        OuterAlgorithm::Newton => Ok(()),
        OuterAlgorithm::FellnerSchall => Err(crate::error::GamrsError::InvalidParameter(format!(
            "{FS_UNAVAILABLE_MARKER} {path} fit path, which runs damped Newton on \
             the REML score instead. Pass method='REML'. Both optimise the same \
             REML criterion and Newton is generally the stronger optimiser here, \
             so this is an optimiser restriction, not a model change."
        ))),
    }
}

impl Default for NewtonOpts {
    fn default() -> Self {
        // Tolerances match mgcv's `gam.fit3.r:1644` defaults — `conv.tol=1e-7`
        // × 5 absolute on the score-relative gradient, `conv.tol=1e-7` on the
        // relative REML change. Mirrors mgcv_rust's smooth.rs:2545-2569 too.
        // Previously set to 1e-9 / 1e-10 (1000× tighter than mgcv) which
        // forced over-convergence — e.g. 10-D Gaussian took 15 outer iters
        // instead of 5-8 because the score plateaus past mgcv-relevant
        // precision. Restoring mgcv parity here keeps `rho_hat` agreement
        // to ~1e-5 (well inside the LinAlg-assembly noise floor) and
        // closes the 10-D Gaussian perf gap from 2.84× to ~1×.
        Self {
            // 200 matches v0.x's `newton_max_iter = max_outer_iter.max(200)`
            // (gam_optimized.rs:1265). Shape-aware multi-smooth fits need
            // O(80-150) iters when a λ_j saturates — at 50 we hit the
            // step-halving fallback and exit with a relaxed tolerance,
            // which produces ρ̂ differences of 1-2 units vs v0.x on the
            // saturated axis. See parity report 2026-05-27.
            max_iters: 200,
            grad_tol: f64::EPSILON.sqrt(),
            reml_tol: 1.0e-7,
            step_min: 1e-3,
            hess_floor: 1e-8,
            max_step: 5.0,
        }
    }
}

/// Damped Newton with Wolfe-style halving. Generic over θ-dim via
/// `ScoreDerivatives` — Phase 0 always has dim=1, but the algorithm is the
/// same shape for the joint (ρ, log σ²) cases Phase 1 brings.
pub struct NewtonWithHalving {
    pub opts: NewtonOpts,
}

impl NewtonWithHalving {
    pub fn new(opts: NewtonOpts) -> Self {
        Self { opts }
    }
}

impl Default for NewtonWithHalving {
    fn default() -> Self {
        Self::new(NewtonOpts::default())
    }
}

impl OuterSolver for NewtonWithHalving {
    fn minimize<S: ScoreDerivatives>(&self, score: &S, theta0: Array1<f64>) -> Result<OuterFit> {
        let opts = &self.opts;
        let axis_caps = score.axis_step_caps();
        let axis_bounds = score.axis_bounds();
        let mut theta = theta0;
        // Clamp the initial point to any per-axis bounds — defensive
        // against `theta0` coming in from a heuristic outside the box.
        if let Some(ref bnds) = axis_bounds {
            for (i, &(lo, hi)) in bnds.iter().enumerate() {
                if i < theta.len() {
                    theta[i] = theta[i].clamp(lo, hi);
                }
            }
        }
        let (mut v, mut g, mut h) = score.value_grad_hess(&theta)?;
        let mut prev_v = f64::INFINITY;

        // Nothing to optimise. An all-parametric design has no penalties and,
        // for a fixed-dispersion family, no shape parameters either — so θ is
        // empty and there is no step to take. Return before the Newton
        // machinery, whose `active.is_empty()` safeguard would otherwise fall
        // back to `vec![0]` and index an empty gradient.
        //
        // This used to be handled incidentally: the old top-of-loop test
        // returned as soon as `grad_norm < tol`, and `inf_norm` of an empty
        // vector is 0. Adding mgcv's score-change VETO removed that accident,
        // because the veto holds convergence off for the first three iterations
        // regardless — so the empty case walked into the step builder and
        // panicked. Caught by `tests/python/test_smoke.py`'s all-parametric
        // Bernoulli case, which the Rust suite does not cover.
        if theta.is_empty() {
            return Ok(OuterFit {
                theta,
                value: v,
                grad_norm: 0.0,
                iterations: 0,
                converged: true,
            });
        }

        for iter in 0..opts.max_iters {
            if let Some(s) = score.stats() {
                s.bump_outer();
            }
            // Score-relative gradient tolerance, matching mgcv gam.fit3.r:1644.
            let score_scale = v.abs() + 1.0;
            let grad_tol_abs = opts.grad_tol * score_scale;
            let grad_norm = inf_norm(&g);
            let grad_converged = grad_norm < grad_tol_abs;
            // Score-change test — a VETO on convergence, never a trigger for
            // it. mgcv `fast-REML.r:1587-1603` sets `converged <- TRUE`, clears
            // it if any |grad| exceeds tolerance, and then clears it AGAIN if
            // the REML value is still moving (re-enabling every axis so it
            // "can't progress" otherwise). It never concludes convergence FROM
            // a small score change.
            //
            // This code had the test inverted: a small `|ΔREML|` returned
            // `converged: true`, so the loop stopped wherever the steps went
            // quiet — which on a flat λ ridge is far from the argmin. Measured
            // on the synthetic flat-ridge fixture: the criterion's argmin is at
            // ρ ≈ 21.92 and the loop was stopping at ρ ≈ 16.77, leaving 3.6e-4
            // REML units and a systematic edf overshoot on every real adjuster
            // term. The gradient test below is the one that decides.
            //
            // NB: this only helps once the criterion is right. With the
            // shipped `log|A|`, running the optimiser further makes the fit
            // WORSE (garage_spaces $577 → $807) because it converges harder
            // onto the wrong function; with the observed `log|H|` the same
            // change gives $183 → $21.
            let reml_still_moving = if iter >= 3 {
                let denom = v.abs().max(1.0);
                ((v - prev_v) / denom).abs() > opts.reml_tol
            } else {
                true
            };

            if grad_converged && !reml_still_moving {
                return Ok(OuterFit {
                    theta,
                    value: v,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                });
            }

            // Newton step with mgcv R's `gam.fit3.r:~1380-1643` stack:
            //   1. Subset Newton: filter axes where either |g_i| or |H_ii|
            //      is meaningfully above the score-relative tolerance. Run
            //      the rest of the Newton machinery on the subset only;
            //      frozen axes get a zero step. Saves work AND keeps an
            //      effectively-dead axis from polluting the active axes'
            //      step direction through the joint inverse.
            //   2. Diagonal preconditioning: D_ii = sqrt(|H_ii|) on the
            //      active sub-Hessian, normalises the eigenvalue spectrum.
            //   3. Gill-Murray-Wright eigen-fix on the preconditioned Hess
            //      (`make_psd_gmw`): ABS negative eigvals + relative floor
            //      at max(|λ|) · ε^0.7. Safe in preconditioned coords.
            //   4. Solve in preconditioned coords, back-transform, pad
            //      frozen dims with zero.
            let dim = g.len();
            // Subset filter — mgcv's `uconv.ind` at gam.fit3.r:1643:
            //   active_i ⇔ |g_i| > dim_tol  OR  |H_ii| > dim_tol
            // mgcv `gam.fit3.r:1643`: `uconv.ind <- (abs(grad) >
            // score.scale*conv.tol*.1) | (abs(grad2) > score.scale*conv.tol*.1)`
            // with `conv.tol = 1e-6` for the scat path (`mgcv.r:2209`), i.e.
            // `1e-7 · score_scale`. The H_ii OR clause keeps axes active when
            // curvature is meaningful even if the gradient is small
            // (saddle-point case).
            //
            // NB this was briefly changed to `score_scale * grad_tol * 0.1` on
            // the belief that the hardcoded 1e-7 was "67× looser than mgcv's".
            // It was not — 1e-7·score_scale IS mgcv's value; the 67× came from
            // comparing against `fast.REML.fit`'s `sqrt(eps)`, which drives the
            // Gaussian fREML path, not scat. Reverted, and it made no
            // measurable difference either way.
            let dim_tol = score_scale * 1.0e-7;
            let active: Vec<usize> = (0..dim)
                .filter(|&i| g[i].abs() > dim_tol || h[[i, i]].abs() > dim_tol)
                .collect();
            // Safeguard (mgcv gam.fit3.r:1432): at least one active axis.
            let active = if active.is_empty() {
                let argmax = (0..dim)
                    .max_by(|&a, &b| g[a].abs().total_cmp(&g[b].abs()))
                    .unwrap_or(0);
                vec![argmax]
            } else {
                active
            };
            let n_active = active.len();
            // Build the active sub-Hessian + sub-gradient.
            let mut diag_precond = Array1::<f64>::zeros(n_active);
            for (ki, &ai) in active.iter().enumerate() {
                diag_precond[ki] = h[[ai, ai]].abs().sqrt().max(opts.hess_floor.sqrt());
            }
            let step = {
                let mut h_sub_pre = Array2::<f64>::zeros((n_active, n_active));
                for (ri, &ai) in active.iter().enumerate() {
                    for (ci, &aj) in active.iter().enumerate() {
                        h_sub_pre[[ri, ci]] = h[[ai, aj]] / (diag_precond[ri] * diag_precond[ci]);
                    }
                }
                let mut g_sub_pre = Array1::<f64>::zeros(n_active);
                for (ki, &ai) in active.iter().enumerate() {
                    g_sub_pre[ki] = g[ai] / diag_precond[ki];
                }
                let h_psd = make_psd_gmw(&h_sub_pre, opts.hess_floor);
                let step_sub_pre = match h_psd.solve(&(-&g_sub_pre)) {
                    Ok(s) => s,
                    Err(_) => -&g_sub_pre / opts.hess_floor.max(1.0),
                };
                // Back-transform AND pad frozen dims with zero.
                let mut step = Array1::<f64>::zeros(dim);
                for (ki, &ai) in active.iter().enumerate() {
                    step[ai] = step_sub_pre[ki] / diag_precond[ki];
                }
                step
            };
            // Cap the step — per-axis caps (mgcv-style, set by shape-aware
            // scores per family) if provided; otherwise the global L_∞ cap.
            let scaled_step = if let Some(ref caps) = axis_caps {
                debug_assert_eq!(caps.len(), step.len(), "axis_step_caps length mismatch");
                // Per-axis: scale the whole step by the tightest binding ratio
                // so direction is preserved (matches mgcv's per-axis-binding
                // shrink — `smooth.r build_outer_search_vector`).
                let mut shrink = 1.0_f64;
                for (i, &si) in step.iter().enumerate() {
                    if si.abs() > caps[i] && si.abs() > 0.0 {
                        shrink = shrink.min(caps[i] / si.abs());
                    }
                }
                &step * shrink
            } else {
                let step_norm = inf_norm(&step);
                if step_norm > opts.max_step {
                    &step * (opts.max_step / step_norm)
                } else {
                    step
                }
            };

            // Step-halving until value decreases. Per-axis bounds (if any)
            // are clamped at each trial point — mgcv-style box-constrained
            // Newton (smooth.r:~1976 lo/hi clamp).
            //
            // Line-search probes use `score.value()` only — the grad/Hess
            // are not needed for accept/reject. After acceptance we refresh
            // `(g, h)` at the accepted point with ONE `value_grad_hess` call.
            // For families where `value()` runs a single inner PIRLS solve
            // and `value_grad_hess()` runs PIRLS + analytic-grad + FD-on-grad
            // Hessian, this drops per-trial cost from ~(2d+1) PIRLS to 1
            // (d = θ-dim; mgcv_rust pattern, `gam_optimized.rs:1390-1547`).
            // mgcv `newton()` uses `maxHalf = 30` unconditionally
            // (`gam.fit3.r:1230`, default set at `mgcv.r:2212`). There is no
            // adaptive cap.
            //
            // This carried one, ported from mgcv_rust `smooth.rs:2741-2772`,
            // whose `stalled → 1 halving` branch collapsed the line search to a
            // SINGLE probe at the full capped Newton step exactly when the REML
            // change had gone quiet — i.e. on a flat λ ridge, precisely where
            // more halving is needed rather than less. It also silently
            // invalidated a diagnostic: sweeping `step_min` from 1e-3 to 1e-10
            // appeared to change nothing, because `max_half = 1` exits the loop
            // before `step_min` can ever bind.
            let max_half = 30;
            let mut alpha = 1.0;
            let mut accepted = false;
            let mut accepted_trial: Option<Array1<f64>> = None;
            let mut accepted_v: f64 = v;
            for _ in 0..max_half {
                let mut trial = &theta + &(&scaled_step * alpha);
                if let Some(ref bnds) = axis_bounds {
                    for (i, &(lo, hi)) in bnds.iter().enumerate() {
                        if i < trial.len() {
                            trial[i] = trial[i].clamp(lo, hi);
                        }
                    }
                }
                if let Some(s) = score.stats() {
                    s.bump_line_search_trial();
                }
                if let Ok(v_trial) = score.value(&trial) {
                    if v_trial.is_finite() && v_trial < v - 1e-10 * v.abs() {
                        accepted_v = v_trial;
                        accepted_trial = Some(trial);
                        accepted = true;
                        break;
                    }
                }
                alpha *= 0.5;
                if alpha < opts.step_min {
                    break;
                }
            }
            if let Some(trial) = accepted_trial {
                // One full eval at the accepted point to refresh (g, h).
                // If this fails (extremely rare — the value probe just
                // succeeded), keep the value-only success and let the next
                // iter retry with stale (g, h) — the next halving will
                // self-correct.
                if let Ok((v_full, g_full, h_full)) = score.value_grad_hess(&trial) {
                    prev_v = v;
                    theta = trial;
                    v = v_full;
                    g = g_full;
                    h = h_full;
                } else {
                    prev_v = v;
                    theta = trial;
                    v = accepted_v;
                    // g, h stale — keep last good ones.
                }
            }

            if !accepted {
                // Step-halving exhausted — Newton's quadratic approximation
                // can't find a strictly-decreasing step from this point.
                // Three distinct convergence cases:
                //
                //   (a) Interior minimum: `|grad|_∞` small relative to the
                //       score scale. Double-precision FD Hessians on flat
                //       regions can't find a strictly-decreasing trial,
                //       but the gradient says we're there.
                //
                //   (b) KKT at active box constraint (strict): the
                //       unconstrained Newton step pushed entirely outside
                //       the box on every axis where the step was
                //       non-trivial, so all movement got clamped to zero.
                //
                //   (c) Projected-gradient KKT (general box-constrained
                //       case): some axes are at active bounds with their
                //       gradient pointing outward (blocked by the box),
                //       and the projected gradient on the remaining
                //       feasible axes is small. The standard KKT condition
                //       for box-constrained optimisation — covers ocat's
                //       saturating-θ ridge where one or more shape axes
                //       sit against the bound but the ρ axes still need
                //       to satisfy a (relaxed) gradient tolerance.
                let kkt_at_boundary = axis_bounds.as_ref().is_some_and(|bnds| {
                    // For each axis with any unconstrained Newton movement,
                    // require that movement to point outside the active
                    // bound. Vacuously true if the raw step is ~0 everywhere
                    // (then case (a) applies).
                    let mut any_movement = false;
                    let mut all_blocked = true;
                    for (i, &(lo, hi)) in bnds.iter().enumerate() {
                        if i >= scaled_step.len() {
                            continue;
                        }
                        let si = scaled_step[i];
                        let eps_at_bound = 1e-12 * (theta[i].abs().max(1.0));
                        let at_lo = (theta[i] - lo).abs() <= eps_at_bound;
                        let at_hi = (hi - theta[i]).abs() <= eps_at_bound;
                        // Only count "real" movement; tiny si is noise.
                        let active_step = si.abs() > 1e-12 * (theta[i].abs().max(1.0));
                        if active_step {
                            any_movement = true;
                            let pushes_out = (at_hi && si > 0.0) || (at_lo && si < 0.0);
                            if !pushes_out {
                                all_blocked = false;
                            }
                        }
                    }
                    any_movement && all_blocked
                });
                // Case (c): the gradient projected onto the feasible
                // directions of the box. For each axis at an active
                // bound with grad pointing OUTWARD (blocked), zero out
                // that component before measuring the norm.
                let proj_grad_small = axis_bounds.as_ref().is_some_and(|bnds| {
                    let mut any_at_bound = false;
                    let mut proj = 0.0_f64;
                    for (i, &gi) in g.iter().enumerate() {
                        let (lo, hi) = bnds
                            .get(i)
                            .copied()
                            .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
                        let eps_at_bound = 1e-9 * (theta[i].abs().max(1.0));
                        let at_lo = (theta[i] - lo).abs() <= eps_at_bound;
                        let at_hi = (hi - theta[i]).abs() <= eps_at_bound;
                        let blocked = (at_hi && gi < 0.0) || (at_lo && gi > 0.0);
                        if at_lo || at_hi {
                            any_at_bound = true;
                        }
                        if !blocked {
                            proj = proj.max(gi.abs());
                        }
                    }
                    // Only declare convergence via projected-grad when at
                    // least one axis sits at an active bound — otherwise
                    // fall back to the unprojected case (a) test below.
                    // Use a tier looser than the unprojected `1e-3` because
                    // FD gradients near a box face are noisier than in the
                    // interior (the active-bound axis acts as a non-smooth
                    // jump in the FD stencil that bleeds into nearby axes).
                    any_at_bound && proj < 1e-1 * (v.abs() + 1.0)
                });
                // Case (d): rank-deficient Hessian KKT. mgcv R's
                // `gam.fit5` step 4 ("at convergence test fundamental
                // rank on balanced version of penalized Hessian. Drop
                // unidentifiable parameters") — analog at the outer
                // Newton level. Eigendecompose H, find the working
                // subspace (eigvals above max(|λ|) · ε^0.7), and check
                // the gradient projected onto that subspace. If small,
                // the gradient mass is entirely along the null
                // direction(s) — which means the score is flat there
                // (a coordinated-shift ridge, the canonical ocat
                // failure mode). The optimiser literally can't make
                // progress in those directions; treat as converged.
                let rank_def_proj_small = {
                    let dim = g.len();
                    // Build the (symmetric) Hessian we're testing. Don't
                    // bother symmetrising — the FD asymmetry is small
                    // enough that eigh on the lower triangle is fine.
                    let eig_result = h.eigh(UPLO::Lower);
                    match eig_result {
                        Ok((eigs, vecs)) => {
                            let max_abs = eigs.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                            let null_thresh = max_abs * f64::EPSILON.powf(0.7); // ~1.5e-11 of max
                                                                                // Project g onto the working subspace
                                                                                // (eigenvectors with |λ| > null_thresh).
                                                                                // |proj_g|_∞ = max_k |u_k^T g| where the max
                                                                                // runs only over working-subspace eigvecs.
                            let mut proj_max = 0.0_f64;
                            for k in 0..dim {
                                if eigs[k].abs() <= null_thresh {
                                    continue;
                                }
                                let uk = vecs.column(k);
                                let utg: f64 = uk.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
                                if utg.abs() > proj_max {
                                    proj_max = utg.abs();
                                }
                            }
                            // Only fire when the Hessian is ACTUALLY
                            // rank-deficient (otherwise case (a) covers it).
                            // Use the same tier-looser threshold as case (c).
                            let has_null = eigs.iter().any(|&e| e.abs() <= null_thresh);
                            has_null && proj_max < 1e-1 * (v.abs() + 1.0)
                        }
                        Err(_) => false,
                    }
                };
                return Ok(OuterFit {
                    theta,
                    value: v,
                    grad_norm,
                    iterations: iter + 1,
                    converged: kkt_at_boundary
                        || proj_grad_small
                        || rank_def_proj_small
                        || grad_norm < 1e-3 * (v.abs() + 1.0),
                });
            }
        }

        Err(GamrsError::NotConverged {
            iters: opts.max_iters,
            grad_norm: inf_norm(&g),
        })
    }
}

fn inf_norm(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
}

/// Wood & Fasiolo (2017) Fellner-Schall update.
///
/// Per smoothing parameter `λ_i`, the multiplicative update is
///
/// ```text
///   λ_new[i] = λ[i] · φ · max(rank[i]/λ[i] − tr(A⁻¹ S_i), ε) / (β' S_i β)
/// ```
///
/// where `A = X' W X + Σ_j λ_j S_j` is the inner (penalised) Hessian
/// and `φ` is the dispersion. Performed in log-space with a per-iter
/// step clamp to keep the update stable.
///
/// Port of mgcv_rust `smooth.rs:fellner_schall_step` (which is mgcv R's
/// `bam(method="fREML")` core update).
#[derive(Debug, Clone, Copy)]
pub struct FellnerSchallOpts {
    /// Max outer iterations before bailing out.
    pub max_iters: usize,
    /// Relative `log λ` change for convergence; `max_i |Δlog λ_i| < tol`.
    pub tol: f64,
    /// Per-iter cap on `|Δlog λ_i|`. mgcv default `3.0`.
    pub log_step_clamp: f64,
    /// Lower / upper bounds on λ (clamped post-update). mgcv default
    /// `(1e-9, 1e7)`.
    pub lambda_bounds: (f64, f64),
}

impl Default for FellnerSchallOpts {
    fn default() -> Self {
        Self {
            max_iters: 200,
            tol: 1e-6,
            log_step_clamp: 3.0,
            lambda_bounds: (1e-9, 1e7),
        }
    }
}

/// Run the Fellner-Schall outer loop. Returns the converged
/// `OuterFit { theta: log_lambda_hat, ... }` so the caller can plug
/// it into the same downstream pipeline as Newton's output.
///
/// `phi_fn` reads the dispersion at the current fit — for Bernoulli /
/// Poisson / NegBin it's `1.0`; for Gaussian / Gamma / IG it's the
/// Pearson φ̂ at convergence.
pub fn fellner_schall_minimize<I, S>(
    inner: &I,
    s_list: &[Array2<f64>],
    rank_s_list: &[usize],
    rho0: Array1<f64>,
    phi_fn: impl Fn(&crate::inner::GaussianInnerFit<S>) -> f64,
    opts: FellnerSchallOpts,
    stats: Option<&crate::stats::FitStats>,
) -> Result<OuterFit>
where
    I: crate::traits::InnerSolver<Fit = crate::inner::GaussianInnerFit<S>>,
    S: crate::inner::LinearSolver,
{
    let m = s_list.len();
    debug_assert_eq!(rho0.len(), m);
    debug_assert_eq!(rank_s_list.len(), m);
    let log_lo = opts.lambda_bounds.0.ln();
    let log_hi = opts.lambda_bounds.1.ln();

    let mut rho = rho0;
    for k in 0..m {
        rho[k] = rho[k].clamp(log_lo, log_hi);
    }
    let mut warm_beta: Option<Array1<f64>> = None;
    let tiny = 1e-10_f64;

    for iter in 0..opts.max_iters {
        if let Some(s) = stats {
            s.bump_outer();
        }
        // Single IRLS step at the warm-started β (mgcv R `bam(method=
        // "fREML")` port — audit `nn_exploring/src/pirls/mod.rs:4078-4097`).
        // FS λ-updates move slowly; one IRLS step from a warm β closes
        // the per-outer-iter convergence error well enough for the FS
        // update to find descent. Avoids the 2-3× redundant inner iters
        // that full-PIRLS would do.
        let fit = inner.fit_single_irls(&rho, warm_beta.as_ref())?;
        if let Some(s) = stats {
            s.record_pirls_call(fit.iterations);
        }
        let phi = phi_fn(&fit);
        let a_inv = fit.a_inv();

        // Compute the FS step per smooth and the max |log change|.
        let mut max_change = 0.0_f64;
        let mut new_rho = rho.clone();
        for i in 0..m {
            let s_i = &s_list[i];
            let lambda_i = rho[i].exp().max(1e-20);
            let rank_i = rank_s_list[i] as f64;
            // tr(A⁻¹ S_i)
            let mut tr = 0.0_f64;
            for r in 0..a_inv.nrows() {
                for c in 0..a_inv.ncols() {
                    tr += a_inv[[r, c]] * s_i[[c, r]];
                }
            }
            // β' S_i β
            let s_beta = s_i.dot(&fit.beta);
            let bsb_raw: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            let bsb = bsb_raw.max(tiny);
            let num_raw = rank_i / lambda_i - tr;
            let num = num_raw.max(tiny);
            let log_ratio_raw = (phi * num / bsb).ln();
            let log_ratio = log_ratio_raw.clamp(-opts.log_step_clamp, opts.log_step_clamp);
            let new_log_lambda = (rho[i] + log_ratio).clamp(log_lo, log_hi);
            max_change = max_change.max((new_log_lambda - rho[i]).abs());
            new_rho[i] = new_log_lambda;
        }
        rho = new_rho;
        warm_beta = Some(fit.beta.clone());

        if max_change < opts.tol {
            return Ok(OuterFit {
                theta: rho,
                value: 0.0, // FS doesn't track the REML value directly
                grad_norm: max_change,
                iterations: iter + 1,
                converged: true,
            });
        }
    }

    // Did not converge — return last state with `converged: false`.
    let _ = warm_beta;
    Ok(OuterFit {
        theta: rho,
        value: 0.0,
        grad_norm: f64::NAN,
        iterations: opts.max_iters,
        converged: false,
    })
}

/// Project the symmetric Hessian to a positive-definite form by
/// flooring each eigenvalue at `floor`. Conservative "floor-only"
/// behaviour: negative eigenvalues get clamped to `floor` (small
/// positive), which produces tiny Newton steps in those directions
/// rather than full-magnitude flipped descent steps.
///
/// For the proper Gill-Murray-Wright recipe (ABS + relative floor),
/// see [`make_psd_gmw`]. GMW requires diagonal preconditioning to be
/// stable across families (without preconditioning, the ABS produces
/// over-large steps that destabilise scat / TDist fits).
///
/// `pub(crate)` so `fit::profile_shape` can share this canonical
/// implementation rather than maintain a duplicate.
pub(crate) fn make_psd(h: &Array2<f64>, floor: f64) -> Array2<f64> {
    let d = h.nrows();
    if d == 0 {
        return h.clone();
    }
    if d == 1 {
        let mut out = h.clone();
        if out[[0, 0]] < floor {
            out[[0, 0]] = floor;
        }
        return out;
    }
    // Symmetrise (kills FD-induced asymmetry that would otherwise make
    // eigh use only the lower triangle).
    let mut sym = h.clone();
    for i in 0..d {
        for j in i + 1..d {
            let avg = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = avg;
            sym[[j, i]] = avg;
        }
    }
    let (eigs, vecs) = match sym.eigh(UPLO::Lower) {
        Ok(p) => p,
        Err(_) => {
            // Fallback: diagonal floor only.
            let mut out = sym;
            for i in 0..d {
                if out[[i, i]] < floor {
                    out[[i, i]] = floor;
                }
            }
            return out;
        }
    };
    let mut floored = Array2::<f64>::zeros((d, d));
    for k in 0..d {
        let lam = eigs[k].max(floor);
        let v = vecs.column(k);
        for i in 0..d {
            for j in 0..d {
                floored[[i, j]] += lam * v[i] * v[j];
            }
        }
    }
    floored
}

/// Gill-Murray-Wright modified Hessian — Practical Optimization (1981)
/// p.107-8, mgcv R `gam.fit3.r:1397-1417`, mgcv_rust `smooth.rs:2686-2712`.
///
/// ```text
///   H = U diag(λ_i) U'                       (eigendecomposition)
///   d_i ← |λ_i|                              (ABS for indefinite spectra)
///   d_i ← max(d_i, max(|λ|) · ε^0.7)         (relative floor for tiny)
///   H_psd = U diag(d_i) U'
/// ```
///
/// The ABS step is what unsticks Newton on indefinite spectra: instead
/// of flooring `−λ_i` to a tiny positive (giving a tiny step), GMW
/// preserves `|λ_i|` magnitude (giving a descent step of the right size).
///
/// **Stability caveat**: GMW takes larger steps in formerly-indefinite
/// directions, which only makes sense AFTER the Hessian has been
/// diagonally preconditioned (so the eigenvalue spectrum is uniform
/// across coordinates). Calling GMW on a raw Hessian with wildly
/// different coordinate scales (e.g. ρ and θ axes for scat) over-steps
/// in the high-curvature directions and breaks convergence. Pair with
/// the preconditioning step in the Newton driver.
///
/// Used by the joint Newton in `outer.rs` (preconditioned) and the
/// ρ-only Newton in `profile_shape.rs` (1-D, so preconditioning is
/// trivial / vacuous).
#[allow(dead_code)]
pub(crate) fn make_psd_gmw(h: &Array2<f64>, floor: f64) -> Array2<f64> {
    let d = h.nrows();
    if d == 0 {
        return h.clone();
    }
    if d == 1 {
        let mut out = h.clone();
        out[[0, 0]] = out[[0, 0]].abs().max(floor);
        return out;
    }
    let mut sym = h.clone();
    for i in 0..d {
        for j in i + 1..d {
            let avg = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = avg;
            sym[[j, i]] = avg;
        }
    }
    let (eigs, vecs) = match sym.eigh(UPLO::Lower) {
        Ok(p) => p,
        Err(_) => {
            let mut out = sym;
            for i in 0..d {
                out[[i, i]] = out[[i, i]].abs().max(floor);
            }
            return out;
        }
    };
    let abs_eigs: Vec<f64> = eigs.iter().map(|e| e.abs()).collect();
    let max_abs = abs_eigs.iter().cloned().fold(0.0_f64, f64::max);
    let low_d = (max_abs * f64::EPSILON.powf(0.7)).max(floor);
    let mut floored = Array2::<f64>::zeros((d, d));
    for k in 0..d {
        let lam = abs_eigs[k].max(low_d);
        let v = vecs.column(k);
        for i in 0..d {
            for j in 0..d {
                floored[[i, j]] += lam * v[i] * v[j];
            }
        }
    }
    floored
}

/// Project the symmetric Hessian onto the positive-definite cone by
/// flooring its eigenvalues at `floor`. Matches mgcv's
/// `gam.fit3.r:1397-1417` approach: if `H = Q diag(λ_i) Q'`, return
/// `Q diag(max(λ_i, floor)) Q'`. Guarantees the Newton step is a descent
/// direction even when the analytic / FD Hessian is indefinite (common
/// at the start of joint-θ scat optimisation where σ² and ν are far from
/// their optima).
///
/// 1-D case (Phase 0 / 1): degenerates to `max(h, floor)`. 2-D+ case
/// (Phase 2 scat): full eigendecomposition.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CoordsKind;
    use ndarray::array;

    /// Test-only quadratic score: V(θ) = ½ (θ - θ*)' M (θ - θ*) with known
    /// minimum at θ*. Used to verify multi-d Newton lands at the optimum.
    struct QuadScore {
        m: Array2<f64>,
        opt: Array1<f64>,
    }
    impl ScoreDerivatives for QuadScore {
        fn dim(&self) -> usize {
            self.opt.len()
        }
        fn coords(&self) -> CoordsKind {
            CoordsKind::Identity
        }
        fn value(&self, theta: &Array1<f64>) -> Result<f64> {
            let d = theta - &self.opt;
            Ok(0.5 * d.dot(&self.m.dot(&d)))
        }
        fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
            let d = theta - &self.opt;
            let g = self.m.dot(&d);
            Ok((0.5 * d.dot(&g), g))
        }
        fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)> {
            let (v, g) = self.value_and_grad(theta)?;
            Ok((v, g, self.m.clone()))
        }
    }

    #[test]
    fn newton_lands_on_2d_quadratic_optimum() {
        let m = array![[2.0, 0.5], [0.5, 1.5]];
        let opt = array![1.3, -0.7];
        let score = QuadScore {
            m: m.clone(),
            opt: opt.clone(),
        };
        let solver = NewtonWithHalving::default();
        let fit = solver.minimize(&score, array![0.0, 0.0]).unwrap();
        for i in 0..2 {
            assert!(
                (fit.theta[i] - opt[i]).abs() < 1e-8,
                "θ[{i}] = {} vs opt {}",
                fit.theta[i],
                opt[i]
            );
        }
        assert!(fit.converged);
    }

    #[test]
    fn psd_fix_lifts_negative_eigenvalues() {
        // Saddle-point Hessian — first eigenvalue negative. PSD-fix
        // should flip it to `floor`, leaving the positive one untouched.
        let h = array![[-1.0, 0.0], [0.0, 2.0]];
        let fixed = make_psd(&h, 1e-8);
        let (eigs, _) = fixed.eigh(UPLO::Lower).unwrap();
        let mut sorted = eigs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1e-8).abs() < 1e-12, "small eig {}", sorted[0]);
        assert!((sorted[1] - 2.0).abs() < 1e-12, "large eig {}", sorted[1]);
    }

    #[test]
    fn fellner_schall_refusal_carries_the_marker_python_matches_on() {
        // `python/gamrs/_fitter.py::FS_UNAVAILABLE_MARKER` is this same
        // string, and the wrapper turns a message carrying it into a warned
        // REML refit. Reword the message freely; drop the marker and the
        // Python fallback silently stops firing, which is why this asserts it.
        set_algorithm_override(OuterAlgorithm::FellnerSchall);
        let err = reject_unsupported_algorithm("test")
            .unwrap_err()
            .to_string();
        clear_algorithm_override();
        assert!(
            err.contains(FS_UNAVAILABLE_MARKER),
            "refusal lost the marker: {err}"
        );
        assert!(!err.contains("  "), "message has a run of spaces: {err}");
    }
}
