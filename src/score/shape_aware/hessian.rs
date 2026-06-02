//! Hessian paths for `ShapeAwareEnvelopeScore`.
//!
//! Entry point is `compute_value_grad_hess_analytical` — one PIRLS solve
//! at θ produces (value, gradient, FrozenBetaCtx); the Hessian then
//! comes from a central FD variant chosen by whether the family
//! supplies `analytic_shape_score_gradient`.
//!
//! `compute_value_grad_hess_rho_only` is the **profile-θ** variant: it
//! returns just the M×M ρ block of the joint Hessian, the ρ-only gradient,
//! and the value — no shape FD probes. Used by the `ProfileShapeNewton`
//! outer solver (port of mgcv_rust `src/smooth.rs:1866-1869`'s "legacy
//! M-dim Newton path" comment, plus the NegBin 1-D profile-θ Newton at
//! lines 3562-3637).

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::inner::{GaussianInnerFit, LinearSolver};
use crate::traits::{Link, Loss, VarianceFn};

use super::super::hess_ift::{
    build_xtwx, compute_dev_grad_beta_working_rss, hess_ift_rho, HessIftCtx,
};
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
            // NegBin / scat / Ocat path: analytic IFT for the M×M ρ block
            // (0 PIRLS solves — port of mgcv_rust
            // `reml_hessian_mgcv_exact_ift` at `src/reml/mod.rs:2511-2813`)
            // plus FD-on-grad along shape axes only (2·n_shape PIRLS solves).
            // For NegBin (n_shape=1) the total drops from `2·d` PIRLS solves
            // to `2`. Matches mgcv_rust's M-dim ρ-Newton + 1-D shape-Newton
            // PIRLS economy (`src/smooth.rs:2383` + `3562-3637`).
            self.hess_via_ift_analytic(theta, &fit, &family, n_shape)?
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

    /// ρ-only `(value, g_ρ, H_ρρ)` at θ for the profile-shape Newton path.
    ///
    /// Port of mgcv_rust's NegBin profile-θ pattern: outer Newton steps
    /// **only ρ**, then a separate 1-D log(θ) Newton runs sequentially.
    /// Citations:
    /// - `src/smooth.rs:2383` — `reml_hessian_mgcv_exact_ift` returns an
    ///   M×M ρ-Hessian for NegBin (no log θ axis).
    /// - `src/smooth.rs:1866-1869` — comment block confirms NegBin /
    ///   Tweedie profile blocks run their 1-D log-θ Newton AFTER the ρ
    ///   step ("`joint_active = false` falls through to the legacy M-dim
    ///   Newton path").
    /// - `src/smooth.rs:3562-3637` — the actual NegBin profile-θ Newton
    ///   block (3 PIRLS for central FD on `dlr/d(log θ)`).
    ///
    /// PIRLS economy: **1 PIRLS solve total**. The full `value_grad_hess`
    /// path costs 1 + 2 + 2·n_shape = 5 PIRLS for NegBin 1-D (n_shape=1,
    /// n_terms=1) and 1 + 2 + 2·n_shape = 5 for NegBin 2-D (n_shape=1,
    /// n_terms=2) — the two extras come from `eval_grad_with_fit`'s
    /// shape-FD fallback (gradient.rs:564-574) and `hess_via_ift_analytic`'s
    /// shape-FD column (hessian.rs:474-487). Both vanish in the ρ-only
    /// path because the shape gradient/Hessian are no longer needed; the
    /// 1-D profile-θ Newton evaluates the REML *value* at log θ ± h
    /// (3 PIRLS) outside this function, matching mgcv_rust:3592-3594's
    /// `nb_eval!(log_theta), nb_eval!(log_theta + h), nb_eval!(log_theta - h)`.
    pub fn compute_value_grad_hess_rho_only(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        use super::super::hess_ift::{
            build_xtwx, compute_dev_grad_beta_working_rss, hess_ift_rho, HessIftCtx,
        };

        let (fit, family) = self.fit_inner_at(theta)?;
        let n_terms = self.s_list.len();
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();
        let value = self.score_value(&fit, &family, &rho_slice);

        // ρ-gradient via the shared envelope helper. Per-term `bsb_j`
        // and `tr(H⁻¹·S_j)` are computed here; φ comes from the active
        // Profile (FixedAtOneProfile for NegBin → φ=1).
        let mut bsb_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut tr_hinv_s_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut bsb_total = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = self.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            let tr_hinv_s_j = fit.trace_a_inv(self.s_list[j].view());
            bsb_per_term.push(bsb_j);
            tr_hinv_s_per_term.push(tr_hinv_s_j);
            bsb_total += rho_slice[j].exp() * bsb_j;
        }
        let tr_hinv_xtwx = fit.p as f64;
        let phi = self
            .profile
            .dispersion(&family.loss, &fit, 1.0, bsb_total, tr_hinv_xtwx, self.mp)
            .unwrap_or(1.0);
        let rho_grad = self.compute_rho_envelope_gradient(
            &fit,
            &family,
            &rho_slice,
            &bsb_per_term,
            &tr_hinv_s_per_term,
            phi,
        );
        let mut g_rho = Array1::<f64>::zeros(n_terms);
        for j in 0..n_terms {
            g_rho[j] = rho_grad[j];
        }

        // Analytic M×M ρ-Hessian via the IFT helper — identical to the
        // ρ block built by `hess_via_ift_analytic` (just no shape FD
        // column). Newton-A path for `use_newton_irls()` families (NegBin
        // here): the IFT Hessian differentiates the same `log|H|` the
        // score formula uses (Newton-W, not Fisher-W).
        let lambda: Vec<f64> = rho_slice.iter().map(|&r| r.exp()).collect();
        let sigma2 = phi; // ScaleParameterMethod::Profile uses φ from above.
        let use_newton = family.loss.use_newton_irls();
        let lazy_tk = if use_newton {
            let prior_w = self
                .prior_weights
                .clone()
                .unwrap_or_else(|| Array1::ones(fit.n));
            let rho_arr = Array1::from(rho_slice.clone());
            let s_total = crate::design::combined_s(&self.s_list, &rho_arr);
            crate::inner::pirls::lazy_tk_kkt_inputs(
                &family,
                &self.y,
                &fit.mu,
                &fit.beta,
                &prior_w,
                &self.x_design,
                &self.s_list,
                &s_total,
                &rho_arr,
            )
        } else {
            None
        };
        let (a_inv_owned, xtwx_owned, dev_grad_beta) = if let Some(ref tk) = lazy_tk {
            let a_inv = tk.a_newton_inv.clone();
            let n = fit.n;
            let mut w_newton = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mu_i = fit.mu[i];
                let var_i = family.variance.variance(mu_i).max(1e-300);
                let g_prime_mu = family.link.d_link_dmu(mu_i);
                let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
                let v_prime = family.variance.d_variance(mu_i);
                let v1n = v_prime / var_i;
                let g_double_prime = family.link.d2_link_dmu(mu_i);
                let g2n = g_double_prime / g_prime_mu;
                let c_resid = self.y[i] - mu_i;
                let alpha_raw = 1.0 + c_resid * (v1n + g2n);
                let alpha = if alpha_raw > 0.0 && alpha_raw.is_finite() {
                    alpha_raw
                } else {
                    1.0
                };
                w_newton[i] = wf * alpha;
            }
            let xtwx_n = build_xtwx(&self.x_design, &w_newton);
            let dev_grad = compute_dev_grad_beta_working_rss(
                &self.x_design,
                &w_newton,
                &fit.working_response,
                &fit.beta,
            );
            (a_inv, xtwx_n, dev_grad)
        } else {
            let a_inv = fit.a_inv();
            let xtwx_f = build_xtwx(&self.x_design, &fit.working_weights);
            let dev_grad = compute_dev_grad_beta_working_rss(
                &self.x_design,
                &fit.working_weights,
                &fit.working_response,
                &fit.beta,
            );
            (a_inv, xtwx_f, dev_grad)
        };
        let ctx = HessIftCtx {
            s_list: &self.s_list,
            lambda: &lambda,
            beta: &fit.beta,
            a_inv: &a_inv_owned,
            xtwx: &xtwx_owned,
            sigma2,
            dev_grad_beta: &dev_grad_beta,
        };
        let h_rho = hess_ift_rho(&ctx);

        Ok((value, g_rho, h_rho))
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
    ///
    /// Kept as a fallback / reference for `hess_via_ift_analytic` parity.
    #[allow(dead_code)]
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

    /// Hybrid Hessian: analytic IFT for the M×M ρ block + FD-on-grad for
    /// shape rows/cols.
    ///
    /// Mathematical decomposition of the joint (ρ, θ_shape) Hessian:
    ///
    /// ```text
    ///   H = [ H_ρρ   H_ρθ ]    H_ρρ : M × M     ← analytic IFT
    ///       [ H_θρ   H_θθ ]    H_θρ = H_ρθ'     ← FD-on-grad probes
    ///                          H_θθ : n_shape × n_shape
    /// ```
    ///
    /// - **H_ρρ** comes from [`super::super::hess_ift::hess_ift_rho`], a
    ///   line-by-line port of mgcv_rust `reml_hessian_mgcv_exact_ift`
    ///   (`src/reml/mod.rs:2511-2813`). Zero PIRLS solves.
    /// - **H_θθ and H_ρθ** come from `2·n_shape` central-FD probes on the
    ///   analytic gradient along the shape axes only. Each probe runs
    ///   PIRLS to convergence at perturbed θ_shape, which captures the
    ///   full β-chain through `dβ/dθ_shape` (envelope theorem keeps the
    ///   shape ρ-gradient exact to O(h) when the shape gradient is
    ///   analytic via `analytic_shape_grad_via_ift`).
    /// - **H_θρ** is filled from the FD-on-grad probes' ρ entries of
    ///   `g_plus / g_minus`, then symmetrised with H_ρθ for numerical
    ///   stability (matches `hess_via_fd_on_grad`'s symmetrise pass).
    ///
    /// Total cost: `2 · n_shape` PIRLS solves per outer-Newton iter, vs
    /// `hess_via_fd_on_grad`'s `2 · (M + n_shape)`. NegBin 1-D (M=1,
    /// n_shape=1) drops from 4 to 2 PIRLS; NegBin 2-D from 6 to 2.
    ///
    /// **A consistency choice**: when the loss is on the Newton-IRLS path
    /// (NegBin / TDist), the IFT helper uses the **Newton A** (NOT Fisher
    /// A from `fit.a_factor`) so it differentiates the SAME `log|H|`
    /// the score formula uses (`score.rs:265-282`). The Newton A is
    /// materialised via `lazy_tk_kkt_inputs` (gradient.rs:314), matching
    /// the analytic shape-gradient path's choice. For Ocat (Fisher==Newton)
    /// `fit.a_factor` is used directly.
    fn hess_via_ift_analytic(
        &self,
        theta: &Array1<f64>,
        fit: &GaussianInnerFit<S>,
        family: &crate::family::Family<L, K, V>,
        n_shape: usize,
    ) -> Result<Array2<f64>> {
        let d = theta.len();
        let n_terms = self.s_list.len();
        debug_assert_eq!(d, n_terms + n_shape);
        let mut h = Array2::<f64>::zeros((d, d));

        // -----------------------------------------------------------------
        // 1) Analytic M×M ρ block via the IFT Hessian helper.
        // -----------------------------------------------------------------
        // Build the score's `λ_j = exp(ρ_j)` and σ². Mirrors the gradient
        // path's `bsb_total` aggregation (gradient.rs:200) and the score's
        // dispersion read (`Profile::dispersion`).
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();
        let lambda: Vec<f64> = rho_slice.iter().map(|&r| r.exp()).collect();
        let mut bsb_total = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = self.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            bsb_total += lambda[j] * bsb_j;
        }
        let sigma2 = self
            .profile
            .dispersion(&family.loss, fit, 1.0, bsb_total, fit.p as f64, self.mp)
            .unwrap_or(1.0);

        // Newton-A path for `use_newton_irls()` families (NegBin, TDist).
        // Falls back to Fisher A for Ocat. Mirrors `analytic_shape_grad_via_ift`
        // (gradient.rs:298-329): the score's `log|H|` is the Newton one for
        // these families, so the IFT Hessian MUST differentiate the SAME A.
        let use_newton = family.loss.use_newton_irls();
        let lazy_tk = if use_newton {
            let prior_w = self
                .prior_weights
                .clone()
                .unwrap_or_else(|| Array1::ones(fit.n));
            let rho_arr = Array1::from(rho_slice.clone());
            let s_total = crate::design::combined_s(&self.s_list, &rho_arr);
            crate::inner::pirls::lazy_tk_kkt_inputs(
                family,
                &self.y,
                &fit.mu,
                &fit.beta,
                &prior_w,
                &self.x_design,
                &self.s_list,
                &s_total,
                &rho_arr,
            )
        } else {
            None
        };

        // Materialise `xtwx` and `a_inv` consistently. Newton path: pull
        // `A⁻¹` from `lazy_tk` (which already built A_newton⁻¹), rebuild
        // `w_newton` to derive `xtwx_newton = X' diag(w_newton) X`. Fisher
        // path: build xtwx from `fit.working_weights` and use `fit.a_inv()`.
        // `dev_grad_beta` then uses the SAME W/working_response pair so
        // the term-2 `(dev_grad)·b2` piece stays consistent.
        let (a_inv_owned, xtwx_owned, dev_grad_beta) = if let Some(ref tk) = lazy_tk {
            let a_inv = tk.a_newton_inv.clone();
            // Newton working weights — port of `lazy_tk_kkt_inputs`'s internal
            // `w_newton` (pirls.rs:171-187). Not exposed on TkKKTInputs so we
            // re-derive here using the same `wf · α` formula with the Fisher
            // fallback when α ≤ 0 (matching pirls.rs:268-271 alpha clamp).
            let n = fit.n;
            let mut w_newton = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mu_i = fit.mu[i];
                let var_i = family.variance.variance(mu_i).max(1e-300);
                let g_prime_mu = family.link.d_link_dmu(mu_i);
                let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
                let v_prime = family.variance.d_variance(mu_i);
                let v1n = v_prime / var_i;
                let g_double_prime = family.link.d2_link_dmu(mu_i);
                let g2n = g_double_prime / g_prime_mu;
                let c_resid = self.y[i] - mu_i;
                let alpha_raw = 1.0 + c_resid * (v1n + g2n);
                let alpha = if alpha_raw > 0.0 && alpha_raw.is_finite() {
                    alpha_raw
                } else {
                    1.0
                };
                w_newton[i] = wf * alpha;
            }
            let xtwx_n = build_xtwx(&self.x_design, &w_newton);
            // Newton normal equation: `X'·W_n·(z_n − Xβ) = λSβ` at converged β.
            // `fit.working_response` was stamped at the Newton fixed point
            // (pirls.rs:567 / :707), so `compute_dev_grad_beta_working_rss`
            // returns `-2·λSβ` here.
            let dev_grad = compute_dev_grad_beta_working_rss(
                &self.x_design,
                &w_newton,
                &fit.working_response,
                &fit.beta,
            );
            (a_inv, xtwx_n, dev_grad)
        } else {
            // Fisher / Ocat path.
            let a_inv = fit.a_inv();
            let xtwx_f = build_xtwx(&self.x_design, &fit.working_weights);
            let dev_grad = compute_dev_grad_beta_working_rss(
                &self.x_design,
                &fit.working_weights,
                &fit.working_response,
                &fit.beta,
            );
            (a_inv, xtwx_f, dev_grad)
        };

        let ctx = HessIftCtx {
            s_list: &self.s_list,
            lambda: &lambda,
            beta: &fit.beta,
            a_inv: &a_inv_owned,
            xtwx: &xtwx_owned,
            sigma2,
            dev_grad_beta: &dev_grad_beta,
        };
        let h_rho = hess_ift_rho(&ctx);
        for i in 0..n_terms {
            for j in 0..n_terms {
                h[[i, j]] = h_rho[[i, j]];
            }
        }

        // -----------------------------------------------------------------
        // 2) FD-on-grad for shape rows/cols (n_shape probes, 2·n_shape PIRLS).
        // -----------------------------------------------------------------
        if n_shape > 0 {
            let eps = 1.0e-4;
            for k in 0..n_shape {
                let mut t_plus = theta.clone();
                let mut t_minus = theta.clone();
                t_plus[n_terms + k] += eps;
                t_minus[n_terms + k] -= eps;
                let (_, g_plus) = self.compute_value_grad(&t_plus)?;
                let (_, g_minus) = self.compute_value_grad(&t_minus)?;
                let col = n_terms + k;
                for j in 0..d {
                    h[[j, col]] = (g_plus[j] - g_minus[j]) / (2.0 * eps);
                }
            }
            // Symmetrise the cross block (n_terms × n_shape) — the FD column
            // for shape axis k carries `H_ρ_k = ∂g_ρ/∂θ_k`; H[i, n_terms+k]
            // was just filled, and H[n_terms+k, i] should equal it.
            for i in 0..n_terms {
                for k in 0..n_shape {
                    let col = n_terms + k;
                    h[[col, i]] = h[[i, col]];
                }
            }
            // Symmetrise within the shape block too (in case the FD probe
            // introduced asymmetry — matches `hess_via_fd_on_grad`'s pass).
            for a in 0..n_shape {
                for b in (a + 1)..n_shape {
                    let r = n_terms + a;
                    let c = n_terms + b;
                    let avg = 0.5 * (h[[r, c]] + h[[c, r]]);
                    h[[r, c]] = avg;
                    h[[c, r]] = avg;
                }
            }
        }

        Ok(h)
    }
}
