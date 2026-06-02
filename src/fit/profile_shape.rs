//! Profile-shape outer-Newton driver — port of mgcv_rust's NegBin pattern.
//!
//! mgcv_rust runs an M-dim ρ-Newton followed by a sequential 1-D log(θ)
//! Newton each outer iter (`src/smooth.rs:1866-1869` comment + lines
//! 3562-3637 implementation). gamrs's standard `fit_shape_aware` joint-
//! Newton path costs ~9 PIRLS solves per outer iter for NegBin (1 center
//! + 2 for shape-grad FD + 6 for FD-on-grad Hessian shape column). mgcv's
//! profile pattern costs ~4 (1 center + 3 for log θ central-FD value
//! probes). This driver matches mgcv_rust's PIRLS economy.
//!
//! **Used by NegBin only today.** Tweedie has a closed-form shape grad
//! (uses `hess_via_fd_frozen_beta`, already cheap), and scat/Ocat use
//! the joint Newton in mgcv_rust too (`joint_tdist_active` /
//! `joint_ocat_active`), so they stay on `fit_shape_aware`.
//!
//! Citations (mgcv_rust at /home/alex/vibe_coding/nn_exploring):
//! - `src/smooth.rs:1866-1869` — architectural comment "NegBin/Tweedie
//!   profile blocks run their 1-D Newton AFTER the ρ step".
//! - `src/smooth.rs:2383` — `reml_hessian_mgcv_exact_ift` returns an
//!   M×M ρ-Hessian (no log θ axis).
//! - `src/smooth.rs:3562-3637` — the NegBin profile-θ block (3 PIRLS for
//!   central FD on `dlr/d(log θ)` + `newton_1d_with_halving`).
//! - `src/reml/search_vector.rs:66-92` — `newton_1d_with_halving` helper
//!   (1-D Newton + 2-step line search halving on `δ = -g/max(|H|,1e-4)`,
//!   clamped to `[-step_cap, +step_cap]`).

use std::marker::PhantomData;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::{Eigh, Solve, UPLO};

use crate::design::PreparedDesign;
use crate::error::Result;
use crate::family::Family;
use crate::inner::{GaussianInnerFit, LinearSolver, PirlsOpts};
use crate::outer::NewtonOpts;
use crate::score::shape_aware::{PirlsInnerBuilder, ShapeAwareEnvelopeScore, ShapeInnerBuilder};
use crate::score::FixedAtOneProfile;
use crate::traits::{CoordsKind, InnerSolver, Link, Loss, OuterFit, ScoreDerivatives, VarianceFn};

use super::{compute_edf, compute_edf_per_term, compute_vcov, FittedGam, LinkKind};

/// Outer-Newton + final-fit + EDF for the **profile-θ** shape-aware stack.
///
/// Matches `fit_shape_aware` (driver.rs:279) at the call surface but uses
/// `ProfileShapeNewton` (M-dim ρ-Newton + 1-D log(θ) profile Newton) instead
/// of `NewtonWithHalving` (joint `[ρ; log θ]` Newton). The ρ-Newton uses
/// the analytic IFT M×M ρ-Hessian via `compute_value_grad_hess_rho_only`
/// (1 PIRLS / outer iter); the log-θ Newton uses central-FD on the REML
/// value (3 PIRLS / outer iter).
///
/// Limited to families with `FixedAtOneProfile` (φ ≡ 1) and exactly **1
/// shape param** in log-space. NegBin's `log θ` is the only fit today.
///
/// `S: LinearSolver` propagates from the inner builder to the emitted fit.
pub(crate) fn fit_shape_aware_profile<L, K, V, S, RF, SF>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    family_base: Family<L, K, V>,
    theta0: Array1<f64>,
    rebuild_final_family: RF,
    scale_fn: SF,
    link_kind: LinkKind,
) -> Result<FittedGam>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver,
    RF: FnOnce(&Array1<f64>) -> Family<L, K, V>,
    SF: FnOnce(&Family<L, K, V>, &GaussianInnerFit<S>, &Array1<f64>) -> f64,
{
    let prior = prior_weights.map(|w| w.to_owned());
    let n = x.nrows();

    let pirls_opts = PirlsOpts {
        dev_rel_tol: family_base.loss.pirls_dev_rel_tol(),
        ..Default::default()
    };
    let score: ShapeAwareEnvelopeScore<L, K, V, PirlsInnerBuilder, FixedAtOneProfile, S> =
        ShapeAwareEnvelopeScore {
            x_design: prep.x_design.clone(),
            y: y.to_owned(),
            prior_weights: prior.clone(),
            s_list: prep.s_list.clone(),
            family_base: family_base.clone(),
            rank_s_list: prep.rank_s_list.clone(),
            mp: prep.mp,
            log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
            coords: CoordsKind::Identity,
            pirls_opts,
            inner_builder: PirlsInnerBuilder,
            profile: FixedAtOneProfile,
            _solver: PhantomData,
            accepted_state: std::cell::RefCell::new(None),
            stats: crate::stats::FitStats::new(),
        };

    let n_terms = prep.s_list.len();
    debug_assert_eq!(
        theta0.len(),
        n_terms + 1,
        "profile-θ driver expects exactly 1 shape axis"
    );

    let solver = ProfileShapeNewton::new(
        crate::outer::resolve_tuning(&score.family_base.loss).to_newton_opts(),
    );
    let outer = solver.minimize(&score, theta0)?;

    // Final fit: rebuild family with shape from `outer.theta[n_terms..]`.
    let rho_hat: Array1<f64> = outer.theta.slice(ndarray::s![..n_terms]).to_owned();
    let family_final = rebuild_final_family(&outer.theta);
    let final_inner = score.inner_builder.build(
        family_final.clone(),
        prep.x_design.clone(),
        y.to_owned(),
        prior,
        prep.s_list.clone(),
        PirlsOpts::default(),
    );
    let final_fit: GaussianInnerFit<S> = final_inner.fit(&rho_hat)?;

    let edf = compute_edf(&prep.x_design, &final_fit.working_weights, &final_fit);
    let scale = scale_fn(&family_final, &final_fit, &outer.theta);
    let vcov = compute_vcov(&final_fit, scale);
    let lambda_vec: Array1<f64> = rho_hat.iter().map(|&r| r.exp()).collect();
    let edf_per_term =
        compute_edf_per_term(&prep.s_list, &rho_hat, prep.x_design.ncols(), &final_fit);

    let shape_params = if outer.theta.len() > n_terms {
        outer.theta.slice(ndarray::s![n_terms..]).to_owned()
    } else {
        Array1::<f64>::zeros(0)
    };

    Ok(FittedGam {
        beta: final_fit.beta,
        rho: rho_hat,
        lambda: lambda_vec,
        scale,
        edf_total: edf,
        edf_per_term,
        n,
        n_iters: outer.iterations,
        converged: outer.converged && final_fit.converged,
        reml_value: outer.value,
        predictor: prep.predictor,
        vcov,
        link_kind,
        shape_params,
        stats: score.stats.snapshot(),
    })
}

/// Outer Newton specialised to the **profile-shape** pattern: M-dim
/// ρ-Newton with line search, followed by a sequential 1-D log(θ) Newton
/// (3 PIRLS for central FD on the REML value) each outer iter. Port of
/// mgcv_rust `src/smooth.rs:3562-3637`'s NegBin profile-θ block (plus
/// the M-dim ρ-Newton above it).
///
/// Specialised to `ShapeAwareEnvelopeScore<L, K, V, PirlsInnerBuilder,
/// FixedAtOneProfile, S>` with exactly 1 shape axis — NegBin today.
/// (TDist/Tweedie/Ocat use `NewtonWithHalving` joint Newton via
/// `fit_shape_aware`.)
pub(crate) struct ProfileShapeNewton {
    pub opts: NewtonOpts,
}

impl ProfileShapeNewton {
    pub fn new(opts: NewtonOpts) -> Self {
        Self { opts }
    }

    /// Solve. Mirrors `NewtonWithHalving::minimize`'s structure (outer.rs:86-258)
    /// but operates on the **ρ-block only** for the Newton step and inserts a
    /// 1-D log(θ) profile Newton after each accepted ρ-step.
    pub fn minimize<L, K, V, S>(
        &self,
        score: &ShapeAwareEnvelopeScore<L, K, V, PirlsInnerBuilder, FixedAtOneProfile, S>,
        theta0: Array1<f64>,
    ) -> Result<OuterFit>
    where
        L: Loss + Clone,
        K: Link + Clone,
        V: VarianceFn + Clone,
        S: LinearSolver,
    {
        let opts = &self.opts;
        let n_terms = score.s_list.len();
        let n_shape = score.family_base.n_shape_params();
        debug_assert_eq!(
            n_shape, 1,
            "ProfileShapeNewton requires exactly 1 shape axis (NegBin's log θ)"
        );
        let dim = n_terms + n_shape;
        debug_assert_eq!(theta0.len(), dim);

        // Per-axis caps / bounds for ρ-block (axis_step_caps / axis_bounds
        // on the score return full-dim vectors; we use only the ρ entries).
        let full_caps = score.axis_step_caps();
        let full_bounds = score.axis_bounds();
        let rho_caps: Option<Vec<f64>> = full_caps
            .as_ref()
            .map(|c| c.iter().take(n_terms).copied().collect());
        let rho_bounds: Option<Vec<(f64, f64)>> = full_bounds
            .as_ref()
            .map(|b| b.iter().take(n_terms).copied().collect());
        let shape_step_cap: f64 = full_caps
            .as_ref()
            .map(|c| c[n_terms])
            // mgcv_rust's NegBinLogTheta step_cap (search_vector.rs:1361).
            .unwrap_or(0.5);
        let shape_lo_hi: (f64, f64) = full_bounds
            .as_ref()
            .map(|b| b[n_terms])
            .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));

        let mut theta = theta0;
        // Clamp initial point to bounds.
        if let Some(ref bnds) = full_bounds {
            for (i, &(lo, hi)) in bnds.iter().enumerate() {
                if i < theta.len() {
                    theta[i] = theta[i].clamp(lo, hi);
                }
            }
        }

        // Initial ρ-only (v, g_ρ, H_ρρ). Also retain the converged inner
        // fit so the shape-axis θ-FD probes and line-search candidate
        // evaluations can reuse β̂ via `score_value_frozen_beta` (mgcv_rust
        // `OuterLinearCache::score_at_theta` PIRLS-economy pattern, port of
        // `src/reml/mod.rs:693-729` + `src/smooth.rs:3592-3594`).
        let (mut v, mut g_rho, mut h_rho, mut fit_center) =
            score.compute_value_grad_hess_rho_only_with_fit(&theta)?;
        let mut prev_v = f64::INFINITY;

        for iter in 0..opts.max_iters {
            score.stats.bump_outer();
            // Score-relative gradient tolerance, matching mgcv gam.fit3.r:1644.
            // Convergence checks against the ρ-gradient only — log θ has its
            // own Newton inside the iter that drives `dlr/d(log θ)` to zero
            // separately (mgcv_rust:3608 `last_negbin_log_theta_grad_abs`).
            let score_scale = v.abs() + 1.0;
            let grad_tol_abs = opts.grad_tol * score_scale;
            let grad_norm = inf_norm_view(&g_rho);
            if grad_norm < grad_tol_abs {
                return Ok(OuterFit {
                    theta,
                    value: v,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                });
            }
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

            // -----------------------------------------------------------
            // (1) ρ-Newton step. M-dim Newton on H_ρρ⁻¹ · g_ρ with PSD-fix
            //     and per-axis cap (matches NewtonWithHalving's structure).
            // -----------------------------------------------------------
            let h_psd = make_psd(&h_rho, opts.hess_floor);
            let step_rho = match h_psd.solve(&(-&g_rho)) {
                Ok(s) => s,
                Err(_) => -&g_rho / opts.hess_floor.max(1.0),
            };
            // Per-axis cap on ρ direction.
            let scaled_step_rho = if let Some(ref caps) = rho_caps {
                let mut shrink = 1.0_f64;
                for (i, &si) in step_rho.iter().enumerate() {
                    if si.abs() > caps[i] && si.abs() > 0.0 {
                        shrink = shrink.min(caps[i] / si.abs());
                    }
                }
                &step_rho * shrink
            } else {
                let step_norm = inf_norm_view(&step_rho);
                if step_norm > opts.max_step {
                    &step_rho * (opts.max_step / step_norm)
                } else {
                    step_rho
                }
            };

            // Line search on the REML value over ρ only (shape stays at
            // current value). mgcv_rust:3050-3120 uses the same pattern —
            // the line-search target is just the REML score, no Armijo
            // angle check on the joint gradient.
            //
            // Two-phase strategy port of mgcv_rust:
            //   Phase A: cheap NoRefresh probes filter out clearly-bad
            //     halvings (returns None on family-support failure; returns
            //     a value otherwise). Cheap = one O(p³) Cholesky + dense
            //     mat-vec, no inner PIRLS iteration.
            //   Phase B: when NoRefresh suggests improvement, VERIFY with
            //     full `compute_value` (one PIRLS). NoRefresh approximation
            //     error in DIFFERENCES (~7 decimal absolute, can be 100%
            //     relative when |Δv| < 1e-7·|v|) means a NoRefresh-only
            //     accept can lock onto phantom improvements at the
            //     optimum. The verify step catches that.
            //
            // Net cost: cheap PIRLS-free probes for rejected halvings;
            // 1 PIRLS for the accepted halving. Vs the prior path (1
            // PIRLS per halving including all 20 rejects), we save N−1
            // PIRLS where N is the number of halvings tried.
            // Adaptive halving cap (mgcv_rust smooth.rs:2741-2772). See
            // outer.rs for rationale.
            let stalled = iter >= 3 && ((v - prev_v).abs() / v.abs().max(1.0) < 1.0e-4);
            let max_half = if stalled {
                1
            } else if grad_norm < 0.1 {
                10
            } else if grad_norm < 1.0 {
                20
            } else {
                30
            };
            let mut alpha = 1.0_f64;
            let mut accepted = false;
            let log_theta_current = theta[n_terms];
            let mut accepted_trial: Option<Array1<f64>> = None;
            for _ in 0..max_half {
                let mut trial = theta.clone();
                for i in 0..n_terms {
                    trial[i] += scaled_step_rho[i] * alpha;
                }
                if let Some(ref bnds) = rho_bounds {
                    for (i, &(lo, hi)) in bnds.iter().enumerate() {
                        trial[i] = trial[i].clamp(lo, hi);
                    }
                }
                // shape param unchanged in trial
                trial[n_terms] = log_theta_current;

                score.stats.bump_line_search_trial();
                // Phase A: cheap NoRefresh probe.
                let v_nr = score.compute_value_no_refresh(&trial);
                let nr_suggests_descent = match v_nr {
                    Some(v_trial) => v_trial.is_finite() && v_trial < v - 1e-10 * v.abs(),
                    None => true, // unknown → fall through to full PIRLS verify
                };
                if !nr_suggests_descent {
                    alpha *= 0.5;
                    if alpha < opts.step_min {
                        break;
                    }
                    continue;
                }

                // Phase B: full PIRLS verify on apparent descent.
                if let Ok(v_full) = score.compute_value(&trial) {
                    if v_full.is_finite() && v_full < v - 1e-10 * v.abs() {
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
                // One full eval at the accepted point to refresh (g, h, fit).
                match score.compute_value_grad_hess_rho_only_with_fit(&trial) {
                    Ok((v_full, g_full, h_full, fit_full)) => {
                        prev_v = v;
                        theta = trial;
                        v = v_full;
                        g_rho = g_full;
                        h_rho = h_full;
                        fit_center = fit_full;
                    }
                    Err(_) => {
                        // Extremely rare: value succeeded but full eval failed.
                        // Roll back — let the next iter retry (stale g/h).
                        accepted = false;
                    }
                }
            }

            if !accepted {
                // No improving ρ-step. Still run the log-θ profile-Newton
                // below — mgcv_rust:3206-3245 has similar fall-through
                // logic that runs the SD fallback / convergence check
                // even when Newton stalls. For our 1-D log θ axis the
                // profile block may still find improvement and unstick.
                let kkt_at_boundary = rho_bounds.as_ref().map_or(false, |bnds| {
                    let mut any_movement = false;
                    let mut all_blocked = true;
                    for (i, &(lo, hi)) in bnds.iter().enumerate() {
                        if i >= scaled_step_rho.len() {
                            continue;
                        }
                        let si = scaled_step_rho[i];
                        let eps_at_bound = 1e-12 * (theta[i].abs().max(1.0));
                        let at_lo = (theta[i] - lo).abs() <= eps_at_bound;
                        let at_hi = (hi - theta[i]).abs() <= eps_at_bound;
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
                return Ok(OuterFit {
                    theta,
                    value: v,
                    grad_norm,
                    iterations: iter + 1,
                    converged: kkt_at_boundary || grad_norm < 1e-3 * (v.abs() + 1.0),
                });
            }

            // -----------------------------------------------------------
            // (2) 1-D log(θ) profile Newton — port of mgcv_rust
            //     src/smooth.rs:3562-3637 + reml/search_vector.rs:66-92
            //     `newton_1d_with_halving`. Central FD on the REML value
            //     at log θ ± h gives (g_lθ, H_lθ); Newton step δ =
            //     -g_lθ / max(|H_lθ|, 1e-4), clamped to [-step_cap,
            //     +step_cap]; 2-step halving on the trial.
            //
            //     PIRLS economy (mgcv_rust pattern): the 3 FD probes
            //     (center, +h, -h) and the 1-2 candidate evaluations all
            //     run on the FROZEN β̂ from the accepted ρ probe — exactly
            //     `OuterLinearCache::score_at_theta` (mgcv_rust
            //     `src/reml/mod.rs:693-729`, called from
            //     `src/smooth.rs:3592-3594` via `dispatch_reml_score_with_family`
            //     which uses cached `(y_local, w_local, xtwx_local)`).
            //     β̂ is refreshed at the top of the next outer iter via
            //     `compute_value_grad_hess_rho_only_with_fit`.
            // -----------------------------------------------------------
            let log_theta = theta[n_terms];
            let h_th: f64 = 1e-3; // mgcv_rust:3569
            let rc = v; // already at accepted ρ
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[n_terms] = (log_theta + h_th).clamp(shape_lo_hi.0, shape_lo_hi.1);
            t_minus[n_terms] = (log_theta - h_th).clamp(shape_lo_hi.0, shape_lo_hi.1);
            let rp_v = score.score_value_frozen_beta(&fit_center, &t_plus);
            let rm_v = score.score_value_frozen_beta(&fit_center, &t_minus);
            if rp_v.is_finite() && rm_v.is_finite() {
                let dlr_dlt = (rp_v - rm_v) / (2.0 * h_th);
                let d2lr_dlt2 = (rp_v - 2.0 * rc + rm_v) / (h_th * h_th);

                // newton_1d_with_halving (search_vector.rs:66-92):
                //   denom = max(|H|, 1e-4)
                //   δ = clamp(-g/denom, [-cap, +cap])
                //   try full δ; on failure try δ/2; else stay.
                let denom = d2lr_dlt2.abs().max(1e-4);
                let delta = (-(dlr_dlt / denom))
                    .max(-shape_step_cap)
                    .min(shape_step_cap);
                let candidate = (log_theta + delta).clamp(shape_lo_hi.0, shape_lo_hi.1);

                let mut new_log_theta = log_theta; // base = no-op
                let mut accepted_theta = false;
                let mut theta_try = theta.clone();
                theta_try[n_terms] = candidate;
                let r_new = score.score_value_frozen_beta(&fit_center, &theta_try);
                if r_new.is_finite() && r_new < rc {
                    new_log_theta = candidate;
                    accepted_theta = true;
                    v = r_new;
                }
                if !accepted_theta {
                    let half = (log_theta + 0.5 * delta).clamp(shape_lo_hi.0, shape_lo_hi.1);
                    theta_try[n_terms] = half;
                    let r_half = score.score_value_frozen_beta(&fit_center, &theta_try);
                    if r_half.is_finite() && r_half < rc {
                        new_log_theta = half;
                        v = r_half;
                    }
                }
                theta[n_terms] = new_log_theta;

                // If log θ changed, the ρ-side Hessian / gradient are now
                // stale (the family's θ changed, so PIRLS-converged β̂(ρ, θ)
                // shifted). Refresh `(v, g_ρ, H_ρρ, fit_center)` at the
                // new θ so the next outer iter's gradient-tolerance check
                // and Newton direction are accurate. mgcv_rust:3611 commits
                // the new log θ via `commit_outer_search_vector`; the next
                // iter top runs PIRLS refresh (smooth.rs:2001-2010), then
                // the gradient/Hessian eval at the refreshed (β, w, z, X'WX).
                if new_log_theta != log_theta {
                    let (v_new, g_new, h_new, fit_new) =
                        score.compute_value_grad_hess_rho_only_with_fit(&theta)?;
                    v = v_new;
                    g_rho = g_new;
                    h_rho = h_new;
                    fit_center = fit_new;
                }
            }
        }

        Err(crate::error::GamrsError::NotConverged {
            iters: opts.max_iters,
            grad_norm: inf_norm_view(&g_rho),
        })
    }
}

fn inf_norm_view(v: &Array1<f64>) -> f64 {
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
        let vk = vecs.column(k);
        for i in 0..d {
            for j in 0..d {
                floored[[i, j]] += lam * vk[i] * vk[j];
            }
        }
    }
    floored
}
