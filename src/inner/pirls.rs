//! Penalised IRLS for canonical-link GLM exponential families.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::{GammonError, Result};
use crate::family::Family;
use crate::traits::{InnerSolver, Link, Loss, VarianceFn};

use super::{
    add_penalty, beta_sbeta, factor_and_solve_with_ridge, halve_until_valid,
    weighted_xt, CholeskySolver, GaussianInnerFit, LinearSolver,
};

/// `crate::traits::InnerSolver` impl for any `Family<L, K, V>` via PIRLS.
///
/// Standard penalised iteratively-reweighted least squares loop:
///
/// ```text
///   loop {
///     z   = η + (y - μ) · g'(μ)           // working response
///     W   = 1 / (V(μ) · g'(μ)²)            // working weights (Fisher info)
///     β   = (X'WX + λS)⁻¹ X'Wz             // backend solve
///     η   = X β
///     μ   = g⁻¹(η)
///     if Δdeviance < tol → done
///   }
/// ```
///
/// Step-halving on β when the deviance increases — same shape as mgcv's
/// `gam.fit3.r:840-890` halving loop. Phase 1 ships the canonical-link
/// Fisher path; non-canonical Newton (full `d²L/dμ²` curvature) is
/// deferred.
///
/// `S: LinearSolver` (default `CholeskySolver`) picks the factorisation
/// backend at the type level — `PirlsInner<L, K, V, LuSolver>` swaps
/// Cholesky for LAPACK LU with no other code changes.
pub struct PirlsInner<L: Loss + Clone, K: Link + Clone, V: VarianceFn + Clone, S: LinearSolver = CholeskySolver> {
    pub x_design: Array2<f64>,
    pub y: Array1<f64>,
    pub prior_weights: Option<Array1<f64>>,
    /// Per-term penalty blocks `Vec<S_j>` of `(p, p)`. The PIRLS loop
    /// assembles `S_total(ρ) = Σ_j exp(ρ_j) · S_j` per call to `fit(ρ)`.
    /// Single-smooth callers pass `vec![S]`; multi-smooth `Additive`
    /// passes one block per term.
    pub s_list: Vec<Array2<f64>>,
    pub family: Family<L, K, V>,
    pub opts: PirlsOpts,
    pub _solver: PhantomData<S>,
}

#[derive(Clone)]
pub struct PirlsOpts {
    pub max_iters: usize,
    pub dev_rel_tol: f64,
    pub halving_steps: usize,
    /// Initial η = μ_init mapped through the link. `None` → family-specific
    /// default (`(y + 0.5) / 2` for Bernoulli; `y` clamped for Poisson).
    pub eta_init: Option<Array1<f64>>,
}

impl Default for PirlsOpts {
    fn default() -> Self {
        Self {
            max_iters: 50,
            dev_rel_tol: 1e-9,
            halving_steps: 10,
            eta_init: None,
        }
    }
}

impl<L: Loss + Clone, K: Link + Clone, V: VarianceFn + Clone, S: LinearSolver> InnerSolver
    for PirlsInner<L, K, V, S>
{
    type Fit = GaussianInnerFit<S>;

    fn fit(&self, rho: &Array1<f64>) -> Result<Self::Fit> {
        debug_assert_eq!(
            rho.len(),
            self.s_list.len(),
            "PirlsInner: rho length {} must equal s_list length {}",
            rho.len(),
            self.s_list.len()
        );
        let s_total = crate::design::combined_s(&self.s_list, rho);
        self.pirls_loop(s_total, rho)
    }
}

impl<L: Loss + Clone, K: Link + Clone, V: VarianceFn + Clone, S: LinearSolver>
    PirlsInner<L, K, V, S>
{
    fn pirls_loop(&self, s_total: Array2<f64>, rho: &Array1<f64>) -> Result<GaussianInnerFit<S>> {
        // `lambda_eff = 1` since `s_total` already absorbs the per-term λ_j;
        // every `λ · S` site below now reads as `1 · s_total`. Kept named
        // for readability of the mgcv-equivalent algebra.
        let lambda = 1.0_f64;
        let _ = rho; // rho is captured here via s_total; explicit token kept for grep clarity
        let n = self.x_design.nrows();
        let p = self.x_design.ncols();
        let prior_w: Array1<f64> = match &self.prior_weights {
            Some(w) => w.clone(),
            None => Array1::ones(n),
        };

        // Initial μ — delegated to the Loss via `Loss::initial_mu`. Default
        // is the Bernoulli-style `(y + ȳ) / 2`; Poisson / Bernoulli / Gamma
        // etc. override for link-domain safety. `opts.eta_init` still
        // overrides everything for caller-controlled starts (e.g.
        // warm-restart from a previous fit).
        let mu_init: Array1<f64> = self.family.loss.initial_mu(self.y.view());
        let mut eta: Array1<f64> = mu_init.iter().map(|&m| self.family.link.link(m)).collect();
        if let Some(e0) = &self.opts.eta_init {
            eta.assign(e0);
        }
        let mut mu: Array1<f64> =
            eta.iter().map(|&e| self.family.link.inverse_link(e)).collect();
        let mut dev = self.compute_deviance(&mu, &prior_w);

        let mut beta = Array1::<f64>::zeros(p);
        let mut a_factor_opt: Option<S::Factorization> = None;
        let mut working_weights = Array1::<f64>::ones(n);
        let mut working_response = self.y.clone();
        let mut converged = false;
        let mut iters_used = 0;
        // Penalised deviance at the current accepted state. Starts at the
        // initial-μ deviance (β=0, β'Sβ=0). Tracked alongside `dev` so the
        // mgcv-exact halving (gam.fit3.r:425) can compare pdev-divergence
        // against the previously-accepted state.
        let mut pdev = dev + lambda * beta_sbeta(&s_total, &beta);

        for it in 0..self.opts.max_iters {
            // PIRLS step: build (z, W), backend-solve for β.
            for i in 0..n {
                let mu_i = mu[i];
                let var_i = self.family.variance.variance(mu_i).max(1e-300);
                let g_prime_mu = self.family.link.d_link_dmu(mu_i);
                let w_i = prior_w[i] / (var_i * g_prime_mu * g_prime_mu);
                working_weights[i] = w_i;
                working_response[i] = eta[i] + (self.y[i] - mu_i) * g_prime_mu;
            }

            let (beta_trial, factor_trial) = {
                let xtw = weighted_xt(&self.x_design, &working_weights);
                let xtwx = xtw.dot(&self.x_design);
                let xtwz = xtw.dot(&working_response);
                let mut a = xtwx;
                add_penalty(&mut a, &s_total, lambda);
                // Phase-5b port — ridged factor used ONLY for β̂; the
                // unridged factor is returned as `a_factor` and feeds
                // log|H| / tr(H⁻¹S). See `gaussian_inner_solve`.
                let (factor, b) = factor_and_solve_with_ridge::<S>(&a, xtwz.view())
                    .map_err(|e| match e {
                        GammonError::SingularSystem(msg) => {
                            GammonError::SingularSystem(format!("PIRLS factor: {msg}"))
                        }
                        other => other,
                    })?;
                (b, factor)
            };

            // mgcv-exact three-guard step-halving (gam.fit3.r:382-441).
            // See `halve_until_valid` for the guard sequence; the validity
            // predicate (`eta_mu_valid`) is generic via the Loss trait so
            // the same halving serves every family in the PIRLS path.
            let pdev_old = pdev;
            let iter_one = it == 0;
            let beta_try0 = beta_trial.clone();
            let eta_try0 = self.x_design.dot(&beta_try0);
            let mu_try0: Array1<f64> =
                eta_try0.iter().map(|&e| self.family.link.inverse_link(e)).collect();
            let dev_try0 = self.compute_deviance(&mu_try0, &prior_w);
            let pdev_try0 = dev_try0 + lambda * beta_sbeta(&s_total, &beta_try0);

            let recompute = |b: &Array1<f64>| {
                let e = self.x_design.dot(b);
                let m: Array1<f64> =
                    e.iter().map(|&ev| self.family.link.inverse_link(ev)).collect();
                let d = self.compute_deviance(&m, &prior_w);
                let pd = d + lambda * beta_sbeta(&s_total, b);
                (e, d, pd, Some(m))
            };
            let is_invalid = |e: &Array1<f64>, m: Option<&Array1<f64>>| -> bool {
                let m = m.expect("PIRLS halving always provides μ");
                !self.eta_mu_valid(e, m)
            };

            let (beta_try, eta_try, dev_try, pdev_try, mu_try_opt, accepted) =
                halve_until_valid(
                    beta_try0,
                    &beta,
                    eta_try0,
                    dev_try0,
                    pdev_try0,
                    Some(mu_try0),
                    pdev_old,
                    iter_one,
                    recompute,
                    is_invalid,
                );

            if accepted {
                let dev_change = (dev - dev_try).abs() / (dev.abs() + 1e-30);
                beta = beta_try;
                eta = eta_try;
                mu = mu_try_opt.expect("PIRLS halving always returns μ");
                a_factor_opt = Some(factor_trial);
                if it > 0 && dev_change < self.opts.dev_rel_tol {
                    converged = true;
                }
                dev = dev_try;
                pdev = pdev_try;
            }
            iters_used = it + 1;
            if !accepted {
                // 100 halvings exhausted and still invalid — bail with the
                // last successful state. (Same behaviour as v0.x's revert.)
                break;
            }
            if converged {
                break;
            }
        }

        // If the loop never accepted a step, factor whatever we have at the
        // current (zero) β so the score still receives a usable factor.
        let a_factor = match a_factor_opt {
            Some(f) => f,
            None => {
                // Rebuild A at the current β and factor it — initial β=0
                // makes this `X' diag(prior) X + λS` for unweighted PIRLS.
                let xtw = weighted_xt(&self.x_design, &prior_w);
                let xtwx = xtw.dot(&self.x_design);
                let mut a = xtwx;
                add_penalty(&mut a, &s_total, lambda);
                let max_diag = a.diag().iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
                for i in 0..p {
                    a[[i, i]] += 1e-12 * max_diag;
                }
                S::factorize(a)?
            }
        };

        // For PIRLS at convergence: rss-like quantity for downstream code is
        // the working-RSS `Σ W·(z - X·β)²`, which mgcv calls `dev_num` (it
        // matches the GLM deviance at convergence for canonical links).
        let mut working_rss = 0.0;
        for i in 0..n {
            let r = working_response[i] - eta[i];
            working_rss += working_weights[i] * r * r;
        }

        // For non-canonical-link families that opt into the Newton IRLS
        // path (`Loss::use_newton_irls() = true`), mgcv evaluates the REML
        // score's `log|H|` term against the **Newton** weight matrix
        // (`wf · α`, no per-row Fisher fallback — negative α stay
        // negative). The Fisher H above remains the right object for
        // PIRLS's stability and for `tr(H⁻¹S)` (v0.x's
        // `system.tr_a`), but `log|H|` switches to the Newton W. Ported
        // from `src/reml/mod.rs:436-459` — closes the InverseGaussian +
        // log mgcv parity gap (~0.22 in log|H| → ~3e-4 → ~3e-5 on μ̂).
        // Tk·KK' / Newton-log|H| paths run with the combined `s_total` —
        // they don't see individual term penalties (94b single-smooth
        // gating: the Newton-IRLS path is wired only for `s_list.len() == 1`
        // families today; multi-smooth Newton-IRLS would need per-term
        // η₁_j derivatives, deferred).
        let log_det_h_override = if self.family.loss.use_newton_irls() {
            self.score_log_det_h_newton(&mu, &prior_w, &s_total)
        } else {
            None
        };

        // Pre-compute Tk·KK' gradient inputs for non-canonical-link
        // families. The score body adds `Σ a1[i] · η₁[i] · sign(w[i]) ·
        // lev_uw[i]` to its ρ-gradient (v0.x `src/reml/mod.rs::
        // reml_gradient_mgcv_exact_ift_inner_at_beta`, lines 2068-2176).
        // For canonical links a1 ≡ 0 by envelope on the W-β chain — we
        // skip the inputs entirely so the score body sees `tk_kkt_inputs
        // = None` and short-circuits.
        let tk_kkt_inputs = if self.family.loss.use_newton_irls() {
            self.compute_tk_kkt_inputs(&mu, &beta, &s_total)
        } else {
            None
        };

        Ok(GaussianInnerFit::<S> {
            beta,
            eta,
            mu,
            working_weights,
            working_response,
            deviance: dev,
            rss: working_rss,
            n,
            p,
            iterations: iters_used,
            converged,
            a_factor,
            log_det_h_override,
            tk_kkt_inputs,
        })
    }

    /// Compute the per-row Tk·KK' bits for non-canonical-link families.
    /// Mirrors v0.x `src/reml/mod.rs::reml_gradient_mgcv_exact_ift_inner_at_beta`
    /// at lines 2073-2107: builds `a1[i]` (the Newton-mode IFT weight
    /// derivative) and the unweighted leverage `lev_uw[i] = x_iᵀ A⁻¹ x_i`.
    ///
    /// **Uses eigendecomposition** on the Newton A (not the configured
    /// `LinearSolver`) because the IG + log path produces a potentially
    /// **indefinite** A — Cholesky and LU both fail on indefinite matrices.
    /// `eigh` handles this; the Newton A is symmetric by construction so
    /// it's the right tool regardless of the score backend.
    ///
    /// `a1[i]` formula (mgcv `gdi.c:2556`, Newton path):
    /// ```text
    ///   α   = 1 + (y-μ)·(V'/V + g''/g')              (PIRLS curvature factor)
    ///   xx  = V''/V - (V'/V)² + g'''/g' - (g''/g')²
    ///   α₁  = (-(V'/V + g''/g') + (y-μ)·xx) / α
    ///   a1  = w·(α₁ - V'/V - 2·g''/g') / g'(μ)
    /// ```
    /// Fisher fallback for `α ≤ 0` (matches `compute_irls_wz`):
    /// ```text
    ///   a1 = -w·(V'/V + 2·g''/g') / g'(μ)
    /// ```
    fn compute_tk_kkt_inputs(
        &self,
        mu: &Array1<f64>,
        beta: &Array1<f64>,
        s_total: &Array2<f64>,
    ) -> Option<super::TkKKTInputs> {
        // s_total already encodes `Σ_j λ_j S_j`; the algebra below treats
        // it as the combined `λS` from the single-smooth derivation.
        let lambda = 1.0_f64;
        use ndarray_linalg::{Eigh, UPLO};
        let n = self.x_design.nrows();
        let p = self.x_design.ncols();
        // Newton weights `w_newton[i] = wf · α` (NO Fisher fallback —
        // negative entries stay negative so the Newton A matches v0.x).
        let mut w_newton = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mu_i = mu[i];
            let var_i = self.family.variance.variance(mu_i).max(1e-300);
            let g_prime_mu = self.family.link.d_link_dmu(mu_i);
            let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
            let v_prime = self.family.variance.d_variance(mu_i);
            let v1n = v_prime / var_i;
            let g_double_prime = self.family.link.d2_link_dmu(mu_i);
            let g2n = g_double_prime / g_prime_mu;
            let c_resid = self.y[i] - mu_i;
            let alpha = 1.0 + c_resid * (v1n + g2n);
            w_newton[i] = wf * alpha;
            if !w_newton[i].is_finite() {
                return None;
            }
        }
        // Build A_newton = X' diag(w_newton) X + λS.
        let mut a_newton = Array2::<f64>::zeros((p, p));
        for k in 0..n {
            let wk = w_newton[k];
            for j in 0..p {
                let xkj_w = self.x_design[[k, j]] * wk;
                for l in 0..p {
                    a_newton[[j, l]] += xkj_w * self.x_design[[k, l]];
                }
            }
        }
        for j in 0..p {
            for l in 0..p {
                a_newton[[j, l]] += lambda * s_total[[j, l]];
            }
        }
        // Symmetrise to clean FP drift before eigh.
        for j in 0..p {
            for l in (j + 1)..p {
                let avg = 0.5 * (a_newton[[j, l]] + a_newton[[l, j]]);
                a_newton[[j, l]] = avg;
                a_newton[[l, j]] = avg;
            }
        }
        // A⁻¹ from eigendecomposition (handles indefinite spectra).
        let (eigs, eigvecs) = match a_newton.eigh(UPLO::Lower) {
            Ok(p) => p,
            Err(_) => return None,
        };
        let mut a_inv = Array2::<f64>::zeros((p, p));
        for k in 0..p {
            let lam_k = eigs[k];
            if !lam_k.is_finite() || lam_k.abs() < 1e-300 {
                return None;
            }
            let inv_lam_k = 1.0 / lam_k;
            for i in 0..p {
                let vi = eigvecs[[i, k]];
                for j in 0..p {
                    a_inv[[i, j]] += inv_lam_k * vi * eigvecs[[j, k]];
                }
            }
        }
        // a1[i]: v0.x `src/reml/mod.rs:2392-2415`. Newton branch uses
        // w_newton[i] (signed).
        let mut a1 = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mu_i = mu[i];
            let var_i = self.family.variance.variance(mu_i).max(1e-300);
            let g_prime_mu = self.family.link.d_link_dmu(mu_i);
            if g_prime_mu.abs() < 1e-12 {
                continue;
            }
            let v_prime = self.family.variance.d_variance(mu_i);
            let v1n = v_prime / var_i;
            let v_double_prime = self.family.variance.d2_variance(mu_i);
            let v2n = v_double_prime / var_i;
            let g_double_prime = self.family.link.d2_link_dmu(mu_i);
            let g2n = g_double_prime / g_prime_mu;
            let g_triple_prime = self.family.link.d3_link_dmu(mu_i);
            let g3n = g_triple_prime / g_prime_mu;
            let c_resid = self.y[i] - mu_i;
            let alpha_raw = 1.0 + c_resid * (v1n + g2n);
            let alpha = if alpha_raw <= 0.0 { 1.0 } else { alpha_raw };
            let xx = v2n - v1n * v1n + g3n - g2n * g2n;
            let alpha1 = (-(v1n + g2n) + c_resid * xx) / alpha;
            a1[i] = w_newton[i] * (alpha1 - v1n - 2.0 * g2n) * g_prime_mu.recip();
        }
        // `lev_uw[i] = x_iᵀ A_newton⁻¹ x_i`.
        let mut lev_uw = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut s = 0.0_f64;
            for j in 0..p {
                let mut acc = 0.0_f64;
                for l in 0..p {
                    acc += a_inv[[j, l]] * self.x_design[[i, l]];
                }
                s += self.x_design[[i, j]] * acc;
            }
            lev_uw[i] = s;
        }
        // `b1 = -λ · A_newton⁻¹ · S · β`, then `eta1 = X · b1`.
        let s_beta = s_total.dot(beta);
        let mut a_inv_s_beta = Array1::<f64>::zeros(p);
        for j in 0..p {
            let mut acc = 0.0_f64;
            for l in 0..p {
                acc += a_inv[[j, l]] * s_beta[l];
            }
            a_inv_s_beta[j] = acc;
        }
        let mut b1 = Array1::<f64>::zeros(p);
        for j in 0..p {
            b1[j] = -lambda * a_inv_s_beta[j];
        }
        let eta1 = self.x_design.dot(&b1);
        // tr(A_newton⁻¹ S). v0.x's `_newton_at_beta` uses this for the
        // gradient's `λ·tr(A⁻¹S)/2` term so it matches the rest of the
        // Tk·KK' machinery (all derived against Newton A).
        let mut tr_a_newton_inv_s = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                tr_a_newton_inv_s += a_inv[[i, j]] * s_total[[j, i]];
            }
        }
        // sign(w) factor; v0.x's gdi.c:856 derivation shows it cancels
        // with diagKKt's |w|, so we use 1.0 everywhere (no sign factor).
        let sign_w = Array1::<f64>::ones(n);
        Some(super::TkKKTInputs {
            a1,
            lev_uw,
            eta1,
            tr_a_newton_inv_s,
            working_weights_sign: sign_w,
        })
    }

    /// Build `A_score = X' diag(W_newton) X + λS` at the converged β and
    /// return `Σ log|λ_i|` via symmetric eigendecomposition. `W_newton` is
    /// the row-wise observed-info weight `wf · α` *without* the per-row
    /// Fisher fallback used by the inner PIRLS step — negative α stay
    /// negative so `A_score` is potentially indefinite. Returns `None` if
    /// any eigenvalue is non-finite or numerically zero (the caller then
    /// falls back to the Fisher H's `log|H|`).
    ///
    /// Mirrors v0.x `src/reml/mod.rs:436-459` and `src/linalg.rs::
    /// log_abs_det_symmetric`. Used only when the loss opts into
    /// `use_newton_irls`; canonical-link families never reach here.
    fn score_log_det_h_newton(
        &self,
        mu: &Array1<f64>,
        prior_w: &Array1<f64>,
        s_total: &Array2<f64>,
    ) -> Option<f64> {
        let lambda = 1.0_f64;
        use ndarray_linalg::{Eigh, UPLO};

        let n = self.x_design.nrows();
        let p = self.x_design.ncols();
        let mut w_newton = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mu_i = mu[i];
            let var_i = self.family.variance.variance(mu_i).max(1e-300);
            let g_prime_mu = self.family.link.d_link_dmu(mu_i);
            let wf = 1.0 / (var_i * g_prime_mu * g_prime_mu);
            let v_prime = self.family.variance.d_variance(mu_i);
            let v1n = v_prime / var_i;
            let g_double_prime = self.family.link.d2_link_dmu(mu_i);
            let g2n = g_double_prime / g_prime_mu;
            let c_resid = self.y[i] - mu_i;
            let alpha = 1.0 + c_resid * (v1n + g2n);
            // No Fisher fallback here — keep sign(alpha). The
            // potentially indefinite A is handled by `eigh` below.
            let w_i = prior_w[i] * wf * alpha;
            if !w_i.is_finite() {
                return None;
            }
            w_newton[i] = w_i;
        }
        // Build A_score = X' diag(W_newton) X + λS.
        let mut a_score = Array2::<f64>::zeros((p, p));
        for k in 0..n {
            let wk = w_newton[k];
            for j in 0..p {
                let xkj_w = self.x_design[[k, j]] * wk;
                for l in 0..p {
                    a_score[[j, l]] += xkj_w * self.x_design[[k, l]];
                }
            }
        }
        for j in 0..p {
            for l in 0..p {
                a_score[[j, l]] += lambda * s_total[[j, l]];
            }
        }
        // Symmetrise defensively against FP drift from the manual loop.
        for j in 0..p {
            for l in (j + 1)..p {
                let avg = 0.5 * (a_score[[j, l]] + a_score[[l, j]]);
                a_score[[j, l]] = avg;
                a_score[[l, j]] = avg;
            }
        }
        let eigs = match a_score.eigh(UPLO::Lower) {
            Ok((eigs, _)) => eigs,
            Err(_) => return None,
        };
        let mut log_det = 0.0_f64;
        for e in eigs.iter() {
            let ae = e.abs();
            if ae < 1e-300 || !ae.is_finite() {
                return None;
            }
            log_det += ae.ln();
        }
        Some(log_det)
    }

    fn compute_deviance(&self, mu: &Array1<f64>, prior_w: &Array1<f64>) -> f64 {
        let mut s = 0.0;
        for i in 0..self.y.len() {
            s += prior_w[i] * self.family.loss.deviance_per_obs(self.y[i], mu[i]);
        }
        s
    }

    /// Generic (η, μ)-validity check — gammon's link-/family-agnostic analogue
    /// of mgcv's `family$valideta` and `family$validmu`. Defined in terms of
    /// the existing trait surface (no new "what family is this" dispatch):
    ///   - η: every entry finite (catches `link(μ)`-divergence; mgcv's
    ///     `binomial()$valideta` for instance accepts any finite η).
    ///   - μ: every entry finite (catches `inverse_link` blowing up).
    ///   - deviance per obs is finite for every (y_i, μ_i) — this is the
    ///     family's own validity statement: Bernoulli's μ ∈ (0, 1) and
    ///     Poisson/Gamma/IG's μ > 0 each emit non-finite deviance outside
    ///     their support (because `log(0)` / division by zero / negative
    ///     log argument). Using the Loss as the validity oracle keeps the
    ///     halving generic over all `Loss + Link + VarianceFn` triples.
    fn eta_mu_valid(&self, eta: &Array1<f64>, mu: &Array1<f64>) -> bool {
        for i in 0..eta.len() {
            if !eta[i].is_finite() || !mu[i].is_finite() {
                return false;
            }
            if !self.family.loss.deviance_per_obs(self.y[i], mu[i]).is_finite() {
                return false;
            }
        }
        true
    }
}
