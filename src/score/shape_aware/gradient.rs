//! Value + gradient paths for `ShapeAwareEnvelopeScore`.
//!
//! `compute_rho_envelope_gradient` is the single source of truth for the
//! per-term ρ-gradient (called from `compute_value_grad`, `eval_grad_with_fit`,
//! `eval_grad_frozen_beta`, and the python-side diagnostic). Every
//! gradient path lives here so the formula stays in one place.

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::family::Family;
use crate::inner::{GaussianInnerFit, LinearSolver};
use crate::traits::{Level1ShapeDerivs, Link, Loss, VarianceFn};

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
    /// SINGLE SOURCE OF TRUTH for the per-term envelope ρ-gradient.
    /// Called from `compute_value_grad`, `eval_grad_with_fit`,
    /// `eval_grad_frozen_beta`, and the python-side diagnostic — every
    /// gradient evaluation path goes through here.
    ///
    /// Formula (at PIRLS-converged β):
    /// ```text
    ///   ∂REML/∂ρ_j = λ_j·β'S_jβ/(2φ)
    ///              + 0.5·λ_j·tr(H⁻¹S_j)                ← Fisher envelope
    ///              + 0.5·∂ridge/∂ρ_j · tr(H⁻¹)         ← ridge derivative
    ///              + 0.5·Tk·KK'_j                        ← β-chain in log|H|
    ///              − 0.5·adj_rank_j
    /// ```
    ///
    /// The two "extra" terms — beyond gamrs's prior simple-envelope form —
    /// are the non-canonical-link / non-Gaussian corrections:
    ///
    /// 1. **`∂ridge/∂ρ_j · tr(H⁻¹) / 2`**: post-penalty `max_diag(A)`
    ///    ridge depends on λ. At `i* = argmax_i |A[i,i]|` (post-pen),
    ///    `∂A[i*,i*]/∂ρ_j = λ_j·S_j[i*,i*]` (W constant by envelope).
    ///
    /// 2. **`Tk·KK'_j / 2`**: W = ½·Dmu2(η) depends on β through η = Xβ,
    ///    so `∂(X'WX)/∂ρ_j` is non-zero via the chain:
    ///    `∂β/∂ρ_j = −λ_j·H⁻¹·S_j·β` (IFT on PIRLS score equation),
    ///    `η₁_j = X·∂β/∂ρ_j`,
    ///    `Tk·KK'_j = Σᵢ (½·Dmu3_i) · η₁_j[i] · h_diag[i]`,
    ///    where `h_diag_i = (X·H⁻¹·X')_ii`.
    ///    Only fires when the Loss supplies `level1_shape_derivatives`
    ///    (ocat does; default Loss impl returns None, so other shape-
    ///    aware families fall back to the pure envelope — matching
    ///    gamrs's documented parity floor for those).
    pub(crate) fn compute_rho_envelope_gradient(
        &self,
        fit: &GaussianInnerFit<S>,
        family: &Family<L, K, V>,
        rho_slice: &[f64],
        bsb_per_term: &[f64],
        tr_hinv_s_per_term: &[f64],
        phi: f64,
    ) -> Vec<f64> {
        let n_terms = self.s_list.len();
        let rank_adj = family.loss.score_rank_adjustment();
        let n = fit.n;
        let p = fit.p;

        // Rebuild A_diag = X'WX_diag + Σ λ_j S_j_diag to find i* (the
        // argmax row used by the post-penalty ridge formula).
        let mut a_diag = Array1::<f64>::zeros(p);
        for c in 0..p {
            let mut xtwx_c = 0.0_f64;
            for i in 0..n {
                let xic = self.x_design[[i, c]];
                xtwx_c += xic * xic * fit.working_weights[i];
            }
            a_diag[c] = xtwx_c;
        }
        for j in 0..n_terms {
            let lambda_j = rho_slice[j].exp();
            for c in 0..p {
                a_diag[c] += lambda_j * self.s_list[j][[c, c]];
            }
        }
        let mut i_star = 0_usize;
        let mut best = a_diag[0].abs();
        for c in 1..p {
            let v = a_diag[c].abs();
            if v > best {
                best = v;
                i_star = c;
            }
        }

        let ridge_scale = 1.0e-5 * (1.0 + (n_terms as f64).sqrt());
        // tr(H⁻¹) — diag of A⁻¹ via the fit's factor.
        let mut id_eye = Array2::<f64>::zeros((p, p));
        for c in 0..p {
            id_eye[[c, c]] = 1.0;
        }
        let tr_h_inv = fit.trace_a_inv(id_eye.view());

        // Tk·KK' contribution — only fires when the Loss supplies
        // `level1_shape_derivatives` (currently ocat). For other shape-
        // aware families we use the pure-envelope formula which is the
        // existing documented parity floor.
        let tk_kkt_per_term: Vec<f64> = if let Some(level1) = family.loss.level1_shape_derivatives(
            self.y.view(),
            fit.eta.view(),
            self.prior_weights.as_ref().map(|w| w.view()),
        ) {
            // h_diag[i] = (X · H⁻¹ · X')_ii. Use the fit factor to solve
            // column-wise: A⁻¹ · X' = column-by-column solve(A, X_i).
            let mut a_inv_xt = Array2::<f64>::zeros((p, n));
            for i in 0..n {
                let xi = self.x_design.row(i).to_owned();
                let col = S::solve(&fit.a_factor, xi.view());
                for r in 0..p {
                    a_inv_xt[[r, i]] = col[r];
                }
            }
            let mut h_diag = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mut s = 0.0_f64;
                for r in 0..p {
                    s += self.x_design[[i, r]] * a_inv_xt[[r, i]];
                }
                h_diag[i] = s;
            }
            // For each j: dβ/dρ_j = -λ_j · H⁻¹ · S_j · β. Then η₁_j = X·dβ/dρ_j.
            let mut tk_kkt = vec![0.0_f64; n_terms];
            for j in 0..n_terms {
                let lambda_j = rho_slice[j].exp();
                let s_beta = self.s_list[j].dot(&fit.beta);
                let dbeta_drho_j: Array1<f64> = {
                    let rhs = s_beta.mapv(|v| -lambda_j * v);
                    S::solve(&fit.a_factor, rhs.view())
                };
                let eta1_j = self.x_design.dot(&dbeta_drho_j);
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += 0.5 * level1.dmu3[i] * eta1_j[i] * h_diag[i];
                }
                tk_kkt[j] = s;
            }
            tk_kkt
        } else {
            vec![0.0_f64; n_terms]
        };

        let mut g = Vec::with_capacity(n_terms);
        for j in 0..n_terms {
            let lambda_j = rho_slice[j].exp();
            let adj_rank_j = ((self.rank_s_list[j] as i32 + rank_adj).max(1)) as f64;
            let d_ridge_d_rho_j = ridge_scale * lambda_j * self.s_list[j][[i_star, i_star]];
            g.push(
                lambda_j * bsb_per_term[j] / (2.0 * phi)
                    + 0.5 * lambda_j * tr_hinv_s_per_term[j]
                    + 0.5 * d_ridge_d_rho_j * tr_h_inv
                    + 0.5 * tk_kkt_per_term[j]
                    - 0.5 * adj_rank_j,
            );
        }
        g
    }

    pub(super) fn compute_value(&self, theta: &Array1<f64>) -> Result<f64> {
        let n_terms = self.s_list.len();
        let (fit, family) = self.fit_inner_at(theta)?;
        let rho_slice = theta.slice(ndarray::s![..n_terms]).to_vec();
        Ok(self.score_value(&fit, &family, &rho_slice))
    }

    pub(super) fn compute_value_grad(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>)> {
        let (fit, family) = self.fit_inner_at(theta)?;
        let n_terms = self.s_list.len();
        let n_shape = family.n_shape_params();
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();
        let v = self.score_value(&fit, &family, &rho_slice);

        // Per-term bsb_j, tr_hinv_s_j → envelope ∂REML/∂ρ_j.
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
        let mut g = Array1::<f64>::zeros(n_terms + n_shape);
        for j in 0..n_terms {
            g[j] = rho_grad[j];
        }

        if n_shape > 0 {
            let n_minus_mp = (fit.n as f64) - (self.mp as f64);
            let dp = fit.deviance + bsb_total;
            // First try the analytic envelope-gradient (Tweedie has one
            // closed-form path; Loss::analytic_shape_score_gradient).
            if let Some(analytic) = family.loss.analytic_shape_score_gradient(
                self.y.view(),
                fit.mu.view(),
                dp,
                n_minus_mp,
                phi,
            ) {
                debug_assert_eq!(
                    analytic.len(),
                    n_shape,
                    "analytic_shape_score_gradient returned wrong length"
                );
                for k in 0..n_shape {
                    g[n_terms + k] = analytic[k];
                }
            } else if let Some(level1) = family
                .loss
                .level1_shape_derivatives(self.y.view(), fit.eta.view(), self.prior_weights.as_ref().map(|w| w.view()))
            {
                // IFT-based analytic θ-gradient — ports v0.x's
                // `reml_grad_ocat_theta_block_analytic` (ocat_joint.rs:123-236)
                // generalised to any Loss that supplies Level-1 derivatives.
                let shape_grad =
                    self.analytic_shape_grad_via_ift(&fit, &family, &level1, n_terms)?;
                debug_assert_eq!(shape_grad.len(), n_shape);
                for k in 0..n_shape {
                    g[n_terms + k] = shape_grad[k];
                }
            } else {
                // FD fallback (no analytic path — scat, NegBin).
                let h = 1.0e-5;
                for k in 0..n_shape {
                    let mut t_plus = theta.clone();
                    let mut t_minus = theta.clone();
                    t_plus[n_terms + k] += h;
                    t_minus[n_terms + k] -= h;
                    let v_plus = self.compute_value(&t_plus)?;
                    let v_minus = self.compute_value(&t_minus)?;
                    g[n_terms + k] = (v_plus - v_minus) / (2.0 * h);
                }
            }
        }
        Ok((v, g))
    }

    /// IFT-based analytic θ-gradient assembly from Level-1 derivatives.
    /// Ports v0.x's `reml_grad_ocat_theta_block_analytic` mathematical core.
    ///
    /// For each θ_k:
    /// - `g_k = 0.5·Σᵢ Dth[i,k] + 0.5·tr(H⁻¹ · ∂H/∂θ_k)`
    /// - `∂H/∂θ_k = X' · diag(½·∂Dmu²/∂θ_k) · X` with chain through β:
    ///   `s_ki = ½·(Dmu2th[i,k] + Dmu3[i] · (X · dβ/dθ_k)[i])`
    /// - `dβ/dθ_k = −H⁻¹ · X' · Dmuth[:,k] / 2` (IFT on score equation).
    /// - `tr(H⁻¹ · dH/dθ_k) ≈ Σᵢ s_ki · h_diag[i]` where `h_diag[i] = X_i' H⁻¹ X_i`.
    fn analytic_shape_grad_via_ift(
        &self,
        fit: &GaussianInnerFit<S>,
        family: &Family<L, K, V>,
        level1: &Level1ShapeDerivs,
        _n_terms_for_layout: usize,
    ) -> Result<Array1<f64>> {
        let n = fit.n;
        let p = fit.p;
        let n_theta = level1.dth.ncols();
        debug_assert_eq!(level1.dth.nrows(), n);
        debug_assert_eq!(level1.dmuth.shape(), level1.dth.shape());
        debug_assert_eq!(level1.dmu2th.shape(), level1.dth.shape());
        debug_assert_eq!(level1.dmu3.len(), n);

        // Newton-A inverse (and per-row Newton leverage) if the family is
        // on the Newton-IRLS path (NegBin, IG, scat). The score-side
        // `log|H|` override uses Newton-A's `Σ log|λ_i|`, so the analytic
        // θ-gradient MUST differentiate the same A — `fit.a_factor` is
        // Fisher-A which doesn't match. mgcv R's `gam.fit4.r:gdi2` does
        // the same: passes Newton observed-info weights into the C IFT
        // routine which differentiates Newton-A's log|H|.
        //
        // When no Newton-A is available (ocat goes through OcatInner with
        // gam.fit5 weights that ARE the Newton convention by construction),
        // fall back to Fisher-A via fit.a_factor — ocat's identity link
        // makes Newton-A ≡ Fisher-A anyway.
        let newton_a_inv: Option<&ndarray::Array2<f64>> =
            fit.tk_kkt_inputs.as_ref().map(|t| &t.a_newton_inv);

        // η-coord conversion of the μ-coord Level-1 derivatives. mgcv R's
        // `gam.fit4.r::dDeta()` (lines 5-78) converts via the link-Jacobian
        // factors before plugging into the C `gdi2` IFT machinery — this
        // is exactly the missing piece for non-identity-link families.
        //
        // Per-row factors (g'(μ) = `d_link_dmu`, etc):
        //   ig1 = 1 / g'(μ) = dμ/dη
        //   g2g = g''(μ) / g'(μ)²
        //   g3g = g'''(μ) / g'(μ)³
        //
        // η-coord derivatives (gam.fit4.r:47-51 + `Deta3` at :49):
        //   Detath  = Dmuth · ig1
        //   Deta2th = Dmu2th · ig1²  −  Dmuth · g2g · ig1
        //   Deta3   = Dmu3 · ig1³ − 3·Dmu2·g2g·ig1² + Dmu·(3·g2g² − g3g)·ig1
        //
        // For identity link (`ig1=1, g2g=g3g=0`) all three collapse to
        // the μ-coord values, preserving ocat behaviour exactly. For log
        // link (`ig1=μ, g2g=−1, g3g=2`) the factors above match the
        // hand-derived chain rule for W = ½·D_ηη.
        //
        // Note: mgcv_rust's `tweedie_joint.rs` plugs μ-coord derivs into
        // the IFT formula directly (no `dDeta` conversion) and its parity
        // tests only check μ predictions, so the bug was never exposed
        // there. gamrs diverges from mgcv_rust here and follows mgcv R.
        let mut ig1 = Array1::<f64>::zeros(n);
        let mut g2g = Array1::<f64>::zeros(n);
        let mut g3g = Array1::<f64>::zeros(n);
        let mut dmu_arr = Array1::<f64>::zeros(n);
        let mut dmu2_arr = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mu_i = fit.mu[i];
            let gp = self.family_base.link.d_link_dmu(mu_i);
            let gpp = self.family_base.link.d2_link_dmu(mu_i);
            let gppp = self.family_base.link.d3_link_dmu(mu_i);
            if gp.abs() < 1e-300 {
                // Defensive — link Jacobian shouldn't vanish at converged μ.
                ig1[i] = 0.0;
                g2g[i] = 0.0;
                g3g[i] = 0.0;
            } else {
                ig1[i] = 1.0 / gp;
                g2g[i] = gpp / (gp * gp);
                g3g[i] = gppp / (gp * gp * gp);
            }
            // Dmu / Dmu2 are not in Level1ShapeDerivs (yet); compute from
            // the Loss directly. Prior weights are NOT applied here — the
            // η-coord derivative formulas use the unweighted base values
            // and the prior_w is already baked into Dmuth/Dmu2th/Dmu3 per
            // the existing ocat convention.
            let wt_i = self.prior_weights.as_ref().map(|w| w[i]).unwrap_or(1.0);
            dmu_arr[i] = wt_i * self.family_base.loss.d_loss_dmu(self.y[i], mu_i);
            dmu2_arr[i] = wt_i * self.family_base.loss.d2_loss_dmu(self.y[i], mu_i);
        }

        // Per-row Deta3[i] = Dmu3·ig1³ − 3·Dmu2·g2g·ig1² + Dmu·(3·g2g² − g3g)·ig1.
        let mut deta3 = Array1::<f64>::zeros(n);
        for i in 0..n {
            let ig1_i = ig1[i];
            let ig1_2 = ig1_i * ig1_i;
            let ig1_3 = ig1_2 * ig1_i;
            let g2g_i = g2g[i];
            let g3g_i = g3g[i];
            deta3[i] = level1.dmu3[i] * ig1_3 - 3.0 * dmu2_arr[i] * g2g_i * ig1_2
                + dmu_arr[i] * (3.0 * g2g_i * g2g_i - g3g_i) * ig1_i;
        }

        // dβ/dθ_k = −H⁻¹ · X' · Detath[:, k] / 2 (η-coord IFT).
        // When `newton_a_inv` is available, use it directly — matches the
        // Newton-A `log|H|` the score formula uses. Otherwise fall back to
        // the Fisher-A factor stored on the fit (ocat's case).
        let mut dbeta_dtheta = Array2::<f64>::zeros((p, n_theta));
        for k in 0..n_theta {
            let mut detath_k = Array1::<f64>::zeros(n);
            for i in 0..n {
                detath_k[i] = level1.dmuth[[i, k]] * ig1[i];
            }
            let rhs: Array1<f64> = self.x_design.t().dot(&detath_k) * 0.5;
            if let Some(a_inv) = newton_a_inv {
                // dβ/dθ = -A_newton⁻¹ · X' · Detath / 2.
                let v: Array1<f64> = a_inv.dot(&rhs);
                for r in 0..p {
                    dbeta_dtheta[[r, k]] = -v[r];
                }
            } else {
                let v = S::solve(&fit.a_factor, rhs.view());
                for r in 0..p {
                    dbeta_dtheta[[r, k]] = -v[r];
                }
            }
        }

        // h_diag[i] = X_i' H⁻¹ X_i. With Newton-A, this is the precomputed
        // `lev_uw` from TkKKTInputs (line `pirls.rs::compute_tk_kkt_inputs`
        // already computed it as `x_iᵀ · A_newton⁻¹ · x_i`).
        let h_diag: Array1<f64> = if let Some(tk) = fit.tk_kkt_inputs.as_ref() {
            tk.lev_uw.clone()
        } else {
            // Fisher path: solve A_inv·X' column-wise and reduce.
            let mut a_inv_xt = Array2::<f64>::zeros((p, n));
            for i in 0..n {
                let xi = self.x_design.row(i).to_owned();
                let col = S::solve(&fit.a_factor, xi.view());
                for r in 0..p {
                    a_inv_xt[[r, i]] = col[r];
                }
            }
            let mut h_diag_local = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mut s = 0.0_f64;
                for r in 0..p {
                    s += self.x_design[[i, r]] * a_inv_xt[[r, i]];
                }
                h_diag_local[i] = s;
            }
            h_diag_local
        };

        // `Σᵢ ∂ls_i/∂θ_k` per shape axis — the `-ls$d1` row of mgcv
        // `gam.fit5.r:1668`. Ocat returns zeros (ls≡0) so the original
        // Level1 client never needed this; NegBin / scat / TDist do.
        //
        // **Must read the PERTURBED family** (caller-provided `family`),
        // NOT `self.family_base`: `family_base` holds the construction-time
        // shape params (e.g. θ=3.0 from `negbin_log(3.0)`) and never sees
        // outer-Newton probes. Without this fix, NegBin's analytic shape
        // gradient on the `log θ` axis used `dls/dθ` at the original θ
        // everywhere — which is *constant in the perturbed θ*, contributing
        // an O(10) error on the shape axis (was 6-23% rel-err on the
        // `negbin_multismooth_analytic_grad_matches_fd` test).
        let sum_dls_dtheta = family.loss.sum_saturated_log_lik_dtheta(
            self.y.view(),
            1.0,
            self.prior_weights.as_ref().map(|w| w.view()),
        );
        debug_assert_eq!(
            sum_dls_dtheta.len(),
            n_theta,
            "sum_saturated_log_lik_dtheta must return n_shape_params entries"
        );

        let mut grad = Array1::<f64>::zeros(n_theta);
        for k in 0..n_theta {
            // Envelope: Σᵢ Dth[i, k] = ∂(D + P)/∂θ_k (no β-chain at converged β).
            let mut sum_dth_k = 0.0_f64;
            for i in 0..n {
                sum_dth_k += level1.dth[[i, k]];
            }

            // tr(H⁻¹ ∂H/∂θ_k) = Σᵢ ½·(Deta2th[i,k] + Deta3[i]·x_db_i)·h_diag[i]
            // where (mgcv R `gam.fit4.r:51`):
            //   Deta2th[i,k] = Dmu2th[i,k]·ig1²[i] − Dmuth[i,k]·g2g[i]·ig1[i]
            // and Deta3 is precomputed above. For identity link this
            // collapses to ½·(Dmu2th + Dmu3·x_db) — preserving ocat exactly.
            let mut trace_term = 0.0_f64;
            for i in 0..n {
                let mut x_db_i = 0.0_f64;
                for j in 0..p {
                    x_db_i += self.x_design[[i, j]] * dbeta_dtheta[[j, k]];
                }
                let ig1_i = ig1[i];
                let deta2th_ki = level1.dmu2th[[i, k]] * ig1_i * ig1_i
                    - level1.dmuth[[i, k]] * g2g[i] * ig1_i;
                let s_ki = 0.5 * (deta2th_ki + deta3[i] * x_db_i);
                trace_term += s_ki * h_diag[i];
            }
            // Subtract Σ ∂ls/∂θ_k — closes the `-ls$d1` gap missing on the
            // ocat-only original derivation (ocat: ls≡0, so the term was
            // never tested). Without this, the NegBin IFT shape gradient
            // ships the wrong sign on the log θ axis.
            grad[k] = 0.5 * sum_dth_k + 0.5 * trace_term - sum_dls_dtheta[k];
        }
        Ok(grad)
    }

    /// Gradient at θ given an already-converged inner fit at θ. Mirrors
    /// `compute_value_grad`'s gradient block (no second PIRLS solve).
    /// Returns the gradient plus a `FrozenBetaCtx` that the FD probes
    /// reuse to skip the per-probe inner solve.
    pub(super) fn eval_grad_with_fit(
        &self,
        theta: &Array1<f64>,
        fit: &GaussianInnerFit<S>,
        family: &Family<L, K, V>,
    ) -> Result<(Array1<f64>, FrozenBetaCtx)> {
        let n_terms = self.s_list.len();
        let n_shape = family.n_shape_params();
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();

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
        let phi_center = self
            .profile
            .dispersion(&family.loss, fit, 1.0, bsb_total, tr_hinv_xtwx, self.mp)
            .unwrap_or(1.0);

        let rho_grad = self.compute_rho_envelope_gradient(
            fit,
            family,
            &rho_slice,
            &bsb_per_term,
            &tr_hinv_s_per_term,
            phi_center,
        );
        let mut g = Array1::<f64>::zeros(n_terms + n_shape);
        for j in 0..n_terms {
            g[j] = rho_grad[j];
        }
        if n_shape > 0 {
            let n_minus_mp = (fit.n as f64) - (self.mp as f64);
            let dp = fit.deviance + bsb_total;
            if let Some(analytic) = family.loss.analytic_shape_score_gradient(
                self.y.view(),
                fit.mu.view(),
                dp,
                n_minus_mp,
                phi_center,
            ) {
                debug_assert_eq!(analytic.len(), n_shape);
                for k in 0..n_shape {
                    g[n_terms + k] = analytic[k];
                }
            } else {
                // FD fallback (TDist, NegBin, ocat): runs PIRLS at θ ± h.
                let h = 1.0e-5;
                for k in 0..n_shape {
                    let mut t_plus = theta.clone();
                    let mut t_minus = theta.clone();
                    t_plus[n_terms + k] += h;
                    t_minus[n_terms + k] -= h;
                    let v_plus = self.compute_value(&t_plus)?;
                    let v_minus = self.compute_value(&t_minus)?;
                    g[n_terms + k] = (v_plus - v_minus) / (2.0 * h);
                }
            }
        }
        Ok((
            g,
            FrozenBetaCtx {
                bsb_per_term,
                tr_hinv_s_per_term,
                bsb_total,
                phi_center,
                n_minus_mp: (fit.n as f64) - (self.mp as f64),
                deviance: fit.deviance,
            },
        ))
    }

    /// Evaluate the analytic envelope gradient at `theta` using a FROZEN
    /// inner fit (β̂, μ̂, tr(H⁻¹S), bsb, deviance from θ_center). The
    /// family is cloned and `set_shape_params(θ[1..])` is called so
    /// `analytic_shape_score_gradient` sees the perturbed shape state.
    ///
    /// **Pre-condition**: `family_base.loss.analytic_shape_score_gradient(
    /// ...) == Some(...)`. Callers (only `hess_via_fd_frozen_beta`)
    /// gate on this — otherwise the per-probe gradient at frozen β̂ is
    /// structurally inconsistent with the FD-on-value gradient used at
    /// θ_center, and Newton stalls. Confirmed in the canonical_api
    /// tests during the v0.x port (2026-05-25).
    pub(super) fn eval_grad_frozen_beta(
        &self,
        theta: &Array1<f64>,
        fit: &GaussianInnerFit<S>,
        ctx: &FrozenBetaCtx,
    ) -> Result<Array1<f64>> {
        let n_terms = self.s_list.len();
        let n_shape = self.family_base.n_shape_params();
        debug_assert_eq!(theta.len(), n_terms + n_shape);
        debug_assert_eq!(ctx.bsb_per_term.len(), n_terms);
        let rho_slice: Vec<f64> = theta.slice(ndarray::s![..n_terms]).to_vec();
        let shape_slice: Vec<f64> = theta.iter().skip(n_terms).copied().collect();

        let mut family = self.family_base.clone();
        if n_shape > 0 {
            family.set_shape_params(&shape_slice);
        }

        // bsb_total at perturbed ρ but frozen-β bsb_j per term.
        let bsb_total: f64 = (0..n_terms)
            .map(|j| rho_slice[j].exp() * ctx.bsb_per_term[j])
            .sum();

        // φ at the perturbed family — OwnedByLossProfile (Tweedie) reads
        // `loss.phi`; FixedAtOneProfile stays at 1. Either way the
        // frozen-fit handles (bsb_per_term, tr_hinv_s_per_term) are reused.
        let phi = self
            .profile
            .dispersion(&family.loss, fit, 1.0, bsb_total, fit.p as f64, self.mp)
            .unwrap_or(ctx.phi_center);

        // Reuse the same envelope ρ-gradient helper so the formula stays
        // in one place (commit message: DRY).
        let rho_grad = self.compute_rho_envelope_gradient(
            fit,
            &family,
            &rho_slice,
            &ctx.bsb_per_term,
            &ctx.tr_hinv_s_per_term,
            phi,
        );
        let mut g = Array1::<f64>::zeros(n_terms + n_shape);
        for j in 0..n_terms {
            g[j] = rho_grad[j];
        }
        if n_shape > 0 {
            let dp = ctx.deviance + bsb_total;
            let analytic = family
                .loss
                .analytic_shape_score_gradient(
                    self.y.view(),
                    fit.mu.view(),
                    dp,
                    ctx.n_minus_mp,
                    phi,
                )
                .expect(
                    "eval_grad_frozen_beta called for a family without \
                     analytic_shape_score_gradient — gate this with \
                     has_analytic_shape_grad in the caller.",
                );
            debug_assert_eq!(analytic.len(), n_shape);
            for k in 0..n_shape {
                g[n_terms + k] = analytic[k];
            }
        }
        Ok(g)
    }
}
