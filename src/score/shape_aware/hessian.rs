//! Hessian paths for `ShapeAwareEnvelopeScore`.
//!
//! Entry point is `compute_value_grad_hess_analytical` — one PIRLS solve
//! at θ produces (value, gradient, FrozenBetaCtx); the Hessian then
//! comes from a central FD variant chosen by whether the family
//! supplies `analytic_shape_score_gradient`.

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::inner::{GaussianInnerFit, LinearSolver};
use crate::traits::{Link, Loss, VarianceFn};

use super::super::profile::Profile;
use super::builder::ShapeInnerBuilder;
use super::score::{FrozenBetaCtx, ShapeAwareEnvelopeScore};

impl<L, K, V, B, P, S> ShapeAwareEnvelopeScore<L, K, V, B, P, S>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S>,
    P: Profile<L>,
    S: LinearSolver,
{
    /// Coupled `(value, grad, hess)` — three-way dispatch chosen at the
    /// per-family level to match mgcv_rust's PIRLS-economy pattern.
    ///
    /// Recipe (per v0.x `src/reml/tweedie_joint.rs::
    /// tweedie_theta_grad_hess_analytic` and mgcv_rust's NegBin path at
    /// `src/smooth.rs:1866-1869` / `3562-3639`): ONE PIRLS solve at
    /// θ_center → value + gradient. Hessian dispatch then matches the
    /// cheapest correct FD pattern given what the family supplies:
    ///
    /// 1. **Closed-form shape-grad** (`Loss::analytic_shape_score_gradient
    ///    = Some`): Tweedie. Hessian via partial-freeze FD on the
    ///    analytic gradient — log-λ row re-converges PIRLS (2·M solves);
    ///    shape rows freeze β̂ (0 solves). Total: 1 + 2·M PIRLS.
    /// 2. **Level-1 IFT shape-grad** (`level1_shape_derivatives = Some`):
    ///    NegBin, scat/TDist, Ocat. The shape gradient is analytic via
    ///    the IFT path (`analytic_shape_grad_via_ift` at gradient.rs:278),
    ///    so FD on `compute_value_grad` is FD-on-analytic — no chained-FD
    ///    noise concern that v0.1's `hess_via_fd_on_value` was guarding
    ///    against (it predates the IFT analytic shape gradient, added in
    ///    commit 85946a1). Total: 1 + 2·d PIRLS solves (vs `on_value`'s
    ///    `1 + 2·d²`). For NegBin d=2 (1-D smooth + 1 shape): 5 vs 9
    ///    PIRLS. For d=3 (2-D smooth + 1 shape): 7 vs 19 — matches the
    ///    "M-dim Newton + cheap shape-side" PIRLS economy mgcv_rust gets
    ///    in `src/smooth.rs:2383` (analytic ρ-Hessian via
    ///    `reml_hessian_mgcv_exact_ift`) plus `3562-3639` (3-evals 1-D
    ///    profile-θ Newton).
    /// 3. **No analytic gradient anywhere**: fall back to direct FD on
    ///    the REML score value. None of gamrs's shipped families hit
    ///    this branch today — kept as a safety net for hypothetical
    ///    Loss impls without `level1_shape_derivatives`.
    ///
    /// `Loss::analytic_shape_score_hessian` is an optional override for
    /// the shape×shape block (defaults `None`; currently no gamrs family
    /// uses it — v0.x's FD-on-analytic-grad converges without it).
    pub(super) fn compute_value_grad_hess_analytical(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let (fit, family) = self.fit_inner_at(theta)?;
        let n_terms = self.s_list.len();
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();
        let value = self.score_value(&fit, &family, &rho_slice);
        let (g_center, ctx) = self.eval_grad_with_fit(theta, &fit, &family)?;

        let n_shape = family.n_shape_params();
        let has_analytic_shape_grad = n_shape == 0
            || family
                .loss
                .analytic_shape_score_gradient(
                    self.y.view(),
                    fit.mu.view(),
                    fit.deviance,
                    1.0,
                    ctx.phi_center,
                )
                .is_some();
        // The IFT analytic shape-gradient path (gradient.rs:278) fires
        // whenever the family supplies `level1_shape_derivatives` — even
        // without the simpler closed-form `analytic_shape_score_gradient`.
        // Check it on a tiny probe (just the per-row Level-1 derivs at the
        // converged η̂) so the dispatch below can route NegBin / scat / Ocat
        // off the slow `hess_via_fd_on_value` path.
        let has_ift_shape_grad = n_shape == 0
            || family
                .loss
                .level1_shape_derivatives(
                    self.y.view(),
                    fit.eta.view(),
                    self.prior_weights.as_ref().map(|w| w.view()),
                )
                .is_some();

        let mut hess = if has_analytic_shape_grad {
            // Tweedie path: 2·M PIRLS + 0 shape solves.
            self.hess_via_fd_frozen_beta(theta, &fit, &ctx)?
        } else if has_ift_shape_grad {
            // NegBin / scat / Ocat path: FD on the analytic envelope+IFT
            // gradient. 2·d PIRLS solves total (vs `on_value`'s 1 + 2·d²).
            // Matches mgcv_rust's M-dim ρ-Newton + 1-D shape-Newton PIRLS
            // economy — see `src/smooth.rs:1866-1869` and `:2383`.
            self.hess_via_fd_on_grad(theta)?
        } else {
            // Safety-net path: direct FD on REML value. No gamrs family
            // hits this today (NB/scat/Ocat all supply level-1 derivs).
            self.hess_via_fd_on_value(theta)?
        };

        // Optional family-supplied closed-form shape×shape block. Lives
        // at hess[n_terms..n_terms+n_shape, n_terms..n_terms+n_shape].
        // Currently `None` for every gamrs family — hook for future ports.
        if n_shape > 0 {
            let dp = fit.deviance + ctx.bsb_total;
            if let Some(block) = family.loss.analytic_shape_score_hessian(
                self.y.view(),
                fit.mu.view(),
                dp,
                ctx.n_minus_mp,
                ctx.phi_center,
            ) {
                debug_assert_eq!(block.shape(), &[n_shape, n_shape]);
                for j in 0..n_shape {
                    for k in 0..n_shape {
                        hess[[n_terms + j, n_terms + k]] = block[[j, k]];
                    }
                }
            }
        }

        Ok((value, g_center, hess))
    }

    /// **Partial-freeze** central FD on the analytic gradient.
    ///
    /// - Log-λ row/column (`i == 0`): re-converge PIRLS at θ ± h — the
    ///   β-chain through `dβ̂/dλ` matters far from the optimum (the
    ///   penalty acts directly on β), and freezing β̂ here would make
    ///   Newton stall (verified by canonical_api::tweedie failing in
    ///   the 2026-05-25 v0.x port). Cost: 2 PIRLS solves for this row.
    ///
    /// - Shape rows/columns (`i ≥ 1`): freeze β̂ at θ_center, evaluate
    ///   the analytic shape-gradient at perturbed shape params. The
    ///   envelope theorem makes this exact in the gradient and O(h) in
    ///   the Hessian — mirrors v0.x `tweedie_theta_grad_hess_analytic`
    ///   (`src/reml/tweedie_joint.rs:347-486`). Cost: 0 PIRLS solves
    ///   for these 2·(d-1) entries; only Wright-series / closed-form
    ///   evaluations per probe.
    ///
    /// Total: **2 PIRLS solves per outer Newton iter** (vs v0.1's 2d
    /// + 1 = 7 for Tweedie d=3). The dropped 4 PIRLS solves per outer
    /// iter are the speedup. The off-diagonal log-λ↔shape Hessian
    /// entries fill in symmetrically from the log-λ row (which is
    /// computed with re-converge, so it's correct).
    fn hess_via_fd_frozen_beta(
        &self,
        theta: &Array1<f64>,
        fit: &GaussianInnerFit<S>,
        ctx: &FrozenBetaCtx,
    ) -> Result<Array2<f64>> {
        let d = theta.len();
        let n_terms = self.s_list.len();
        let mut hess = Array2::<f64>::zeros((d, d));
        // Re-converge for each log-λ direction (i ∈ 0..n_terms).
        let eps_rho = 1.0e-4;
        for i in 0..n_terms {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps_rho;
            t_minus[i] -= eps_rho;
            let (_, g_plus_rho) = self.compute_value_grad(&t_plus)?;
            let (_, g_minus_rho) = self.compute_value_grad(&t_minus)?;
            for j in 0..d {
                hess[[j, i]] = (g_plus_rho[j] - g_minus_rho[j]) / (2.0 * eps_rho);
            }
        }
        // Frozen-β̂ for shape directions (i ∈ n_terms..d).
        let eps_shape = 1.0e-5;
        for i in n_terms..d {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps_shape;
            t_minus[i] -= eps_shape;
            let g_plus = self.eval_grad_frozen_beta(&t_plus, fit, ctx)?;
            let g_minus = self.eval_grad_frozen_beta(&t_minus, fit, ctx)?;
            for j in 0..d {
                hess[[j, i]] = (g_plus[j] - g_minus[j]) / (2.0 * eps_shape);
            }
        }
        // Symmetrise — off-diagonal log-λ↔shape gets a clean average.
        for i in 0..d {
            for j in i + 1..d {
                let avg = 0.5 * (hess[[i, j]] + hess[[j, i]]);
                hess[[i, j]] = avg;
                hess[[j, i]] = avg;
            }
        }
        Ok(hess)
    }

    /// Direct central FD of the REML score value (no chained FD through
    /// the gradient). Mirrors v0.x's `reml_joint_ocat_finite_diff`
    /// (`src/smooth.rs:622-694`) for families without
    /// `analytic_shape_score_gradient` (scat / negbin / ocat).
    ///
    /// Diagonal: `(s(θ+h·eᵢ) − 2·s(θ) + s(θ−h·eᵢ)) / h²`.
    /// Off-diagonal: `(s(θ+h·eᵢ+h·eⱼ) − s(θ+h·eᵢ−h·eⱼ)
    ///               − s(θ−h·eᵢ+h·eⱼ) + s(θ−h·eᵢ−h·eⱼ)) / (4 h²)`.
    ///
    /// Cost: `1 + 2d + 2·d(d-1)` score evaluations (each a full PIRLS).
    /// For d=4 (ocat with 2 smooths) that's 33 PIRLS solves — heavy,
    /// but matches v0.x exactly and removes the chained-FD noise that
    /// made gamrs's outer Newton drift on the saturated-λ axis (parity
    /// report 2026-05-27).
    fn hess_via_fd_on_value(&self, theta: &Array1<f64>) -> Result<Array2<f64>> {
        let d = theta.len();
        let mut h = Array2::<f64>::zeros((d, d));
        let eps = 1.0e-4;
        let s0 = self.compute_value(theta)?;
        // Cache one-axis perturbations — reused for both the diagonal and
        // each off-diagonal mixed-difference.
        let mut s_plus = vec![0.0_f64; d];
        let mut s_minus = vec![0.0_f64; d];
        for i in 0..d {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps;
            t_minus[i] -= eps;
            s_plus[i] = self.compute_value(&t_plus)?;
            s_minus[i] = self.compute_value(&t_minus)?;
            h[[i, i]] = (s_plus[i] - 2.0 * s0 + s_minus[i]) / (eps * eps);
        }
        // Off-diagonal mixed central differences.
        for i in 0..d {
            for j in i + 1..d {
                let mut t_pp = theta.clone();
                let mut t_pm = theta.clone();
                let mut t_mp = theta.clone();
                let mut t_mm = theta.clone();
                t_pp[i] += eps;
                t_pp[j] += eps;
                t_pm[i] += eps;
                t_pm[j] -= eps;
                t_mp[i] -= eps;
                t_mp[j] += eps;
                t_mm[i] -= eps;
                t_mm[j] -= eps;
                let s_pp = self.compute_value(&t_pp)?;
                let s_pm = self.compute_value(&t_pm)?;
                let s_mp = self.compute_value(&t_mp)?;
                let s_mm = self.compute_value(&t_mm)?;
                let off = (s_pp - s_pm - s_mp + s_mm) / (4.0 * eps * eps);
                h[[i, j]] = off;
                h[[j, i]] = off;
            }
        }
        Ok(h)
    }

    /// Central FD on the gradient with FULL PIRLS re-converge at each
    /// ±h probe. Active for families whose `compute_value_grad` returns
    /// an analytic gradient end-to-end (NegBin / scat / Ocat via the IFT
    /// path at `gradient.rs:278`). Cost: `2·d` PIRLS solves per outer
    /// Newton iter (vs `hess_via_fd_on_value`'s `1 + 2·d²`). Tweedie
    /// uses the cheaper `hess_via_fd_frozen_beta` instead (closed-form
    /// shape-grad bypasses the shape-row PIRLS solves entirely).
    fn hess_via_fd_on_grad(&self, theta: &Array1<f64>) -> Result<Array2<f64>> {
        let d = theta.len();
        let mut h = Array2::<f64>::zeros((d, d));
        let eps = 1.0e-4;
        for i in 0..d {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps;
            t_minus[i] -= eps;
            let (_, g_plus) = self.compute_value_grad(&t_plus)?;
            let (_, g_minus) = self.compute_value_grad(&t_minus)?;
            for j in 0..d {
                h[[j, i]] = (g_plus[j] - g_minus[j]) / (2.0 * eps);
            }
        }
        for i in 0..d {
            for j in i + 1..d {
                let avg = 0.5 * (h[[i, j]] + h[[j, i]]);
                h[[i, j]] = avg;
                h[[j, i]] = avg;
            }
        }
        Ok(h)
    }
}
