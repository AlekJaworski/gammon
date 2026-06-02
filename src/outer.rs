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

impl Default for NewtonOpts {
    fn default() -> Self {
        // Tolerances match mgcv's `gam.fit3.r:1644` defaults — `conv.tol=1e-7`
        // × 5 absolute on the score-relative gradient, `conv.tol=1e-7` on the
        // relative REML change. Tightening these to e.g. 1e-10 / 1e-12
        // doesn't close the residual ~1e-4 ρ̂ gap on hard fixtures (e.g.
        // low_signal_n1000): gamrs is already at a local minimum of its OWN
        // score at that point — the gap to mgcv comes from tiny linear-
        // algebra-assembly-order differences that shift the zero-gradient
        // location by ~1e-12 in tr(A⁻¹S), which scales up to ~1e-4 in ρ̂
        // and ~1e-6 in predictions on ill-conditioned (large λ̂) cases.
        // Closing it further is its own workstream — port mgcv's exact
        // assembly order or use a rotated-Cholesky path.
        Self {
            // 200 matches v0.x's `newton_max_iter = max_outer_iter.max(200)`
            // (gam_optimized.rs:1265). Shape-aware multi-smooth fits need
            // O(80-150) iters when a λ_j saturates — at 50 we hit the
            // step-halving fallback and exit with a relaxed tolerance,
            // which produces ρ̂ differences of 1-2 units vs v0.x on the
            // saturated axis. See parity report 2026-05-27.
            max_iters: 200,
            grad_tol: 1.0e-9,
            reml_tol: 1.0e-10,
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

        for iter in 0..opts.max_iters {
            if let Some(s) = score.stats() {
                s.bump_outer();
            }
            // Score-relative gradient tolerance, matching mgcv gam.fit3.r:1644.
            let score_scale = v.abs() + 1.0;
            let grad_tol_abs = opts.grad_tol * score_scale;
            let grad_norm = inf_norm(&g);
            if grad_norm < grad_tol_abs {
                return Ok(OuterFit {
                    theta,
                    value: v,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                });
            }
            // REML-change tolerance (active iter ≥ 3, mgcv-style).
            if iter >= 3 {
                let denom = v.abs().max(1.0);
                let reml_change = ((v - prev_v) / denom).abs();
                if reml_change < opts.reml_tol {
                    return Ok(OuterFit {
                        theta,
                        value: v,
                        grad_norm,
                        iterations: iter,
                        converged: true,
                    });
                }
            }

            // Newton step. Replace negative / tiny eigenvalues of H with
            // `hess_floor` so the step direction is always descent — same
            // strategy as mgcv `gam.fit3.r:1397-1417`.
            let h_psd = make_psd(&h, opts.hess_floor);
            let step = match h_psd.solve(&(-&g)) {
                Ok(s) => s,
                Err(_) => -&g / opts.hess_floor.max(1.0), // fallback: steepest descent
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
            let mut alpha = 1.0;
            let mut accepted = false;
            let mut accepted_trial: Option<Array1<f64>> = None;
            let mut accepted_v: f64 = v;
            for _ in 0..20 {
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
                // Two distinct convergence cases:
                //
                //   (a) Interior minimum: `|grad|_∞` small relative to the
                //       score scale. Double-precision FD Hessians on flat
                //       regions can't find a strictly-decreasing trial,
                //       but the gradient says we're there.
                //
                //   (b) KKT at active box constraint: the unconstrained
                //       Newton step pushed entirely outside the box on
                //       every axis where the step was non-trivial, so all
                //       movement got clamped to zero. This is the
                //       saturating-λ ridge: gradient stays bounded away
                //       from zero, but axis bounds make the boundary a
                //       constrained local optimum. (Score-relative
                //       criterion (a) used to fire by accident here once
                //       `|v|` drifted large enough.)
                let kkt_at_boundary = axis_bounds.as_ref().map_or(false, |bnds| {
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
                            let pushes_out =
                                (at_hi && si > 0.0) || (at_lo && si < 0.0);
                            if !pushes_out {
                                all_blocked = false;
                            }
                        }
                    }
                    any_movement && all_blocked
                });
                return Ok(OuterFit {
                    theta,
                    value: v,
                    grad_norm,
                    iterations: iter + 1,
                    converged: kkt_at_boundary
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

fn make_psd(h: &Array2<f64>, floor: f64) -> Array2<f64> {
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
        // floored += lam · v vᵀ
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
}
