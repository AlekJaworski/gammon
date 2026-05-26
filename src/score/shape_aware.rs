//! Phase 2 — Shape-aware envelope score for stateful Loss families.
//!
//! Single generic impl `ShapeAwareEnvelopeScore<L, K, V, B, P, S>` covers
//! both the PIRLS-driven families (TDist/scat, Tweedie, NegBin, Gamma in
//! shape-aware mode) AND the ocat family. The difference is purely in
//! WHICH inner solver to instantiate per probe — `ShapeInnerBuilder`
//! lifts that from a per-family duplicated body into a one-line
//! type-level choice.
//!
//! `S: LinearSolver` (default `CholeskySolver`) propagates the backend
//! choice from the inner-builder through to the score body's
//! `inner_fit.log_det_a()` / `inner_fit.trace_a_inv(...)` calls.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::family::{Family, IdentityLink, OcatLoss, OcatVariance};
use crate::inner::{
    CholeskySolver, GaussianInnerFit, LinearSolver, OcatInner, PirlsInner, PirlsOpts,
};
use crate::traits::{CoordsKind, InnerSolver, Link, Loss, ScoreDerivatives, VarianceFn};

use super::profile::{FixedAtOneProfile, OwnedByLossProfile, Profile};

/// Frozen-β̂ context shared between `eval_grad_with_fit` and
/// `eval_grad_frozen_beta`. Holds the converged-inner quantities that
/// stay constant across ±h shape probes — `bsb`, `tr(H⁻¹S)`, `deviance`,
/// `phi_center`, `n_minus_mp`. The fit itself (β̂, μ̂, factor, p, n) is
/// passed alongside.
#[derive(Clone)]
struct FrozenBetaCtx {
    bsb: f64,
    tr_hinv_s: f64,
    phi_center: f64,
    n_minus_mp: f64,
    deviance: f64,
}

/// Build an `InnerSolver` from a freshly shape-synced family + the score's
/// owned design fields. The score body uses this to rebuild the inner per
/// outer probe (shape params shift every probe; the inner must see the
/// current family).
///
/// Two concrete impls in gammon:
/// - `PirlsInnerBuilder<S>` (generic over `L, K, V, S`) — drives TDist/scat,
///   Tweedie, NegBin via the standard PIRLS loop.
/// - `OcatInnerBuilder<S>` — drives the ocat extended family via `OcatInner`,
///   constrained to `Family<OcatLoss, IdentityLink, OcatVariance>` (the
///   only valid Loss/Link/Variance triple for ordered categorical).
pub trait ShapeInnerBuilder<L: Loss + Clone, K: Link + Clone, V: VarianceFn + Clone, S: LinearSolver = CholeskySolver> {
    type Inner: InnerSolver<Fit = GaussianInnerFit<S>>;

    fn build(
        &self,
        family: Family<L, K, V>,
        x_design: Array2<f64>,
        y: Array1<f64>,
        prior_weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
        opts: PirlsOpts,
    ) -> Self::Inner;
}

/// Unit-struct builder for `PirlsInner<L, K, V, S>`. The `S` parameter
/// is resolved from the score's `ShapeAwareEnvelopeScore<…, S>` type
/// context — the builder itself stays stateless.
#[derive(Clone, Copy, Default)]
pub struct PirlsInnerBuilder;

impl<L, K, V, S> ShapeInnerBuilder<L, K, V, S> for PirlsInnerBuilder
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver,
{
    type Inner = PirlsInner<L, K, V, S>;

    fn build(
        &self,
        family: Family<L, K, V>,
        x_design: Array2<f64>,
        y: Array1<f64>,
        prior_weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
        opts: PirlsOpts,
    ) -> Self::Inner {
        PirlsInner {
            x_design,
            y,
            prior_weights,
            s_list,
            family,
            opts,
            _solver: PhantomData,
        }
    }
}

/// Unit-struct builder for `OcatInner<S>`. The `S` parameter is resolved
/// from the score's `ShapeAwareEnvelopeScore<…, S>` type context.
#[derive(Clone, Copy, Default)]
pub struct OcatInnerBuilder;

impl<S: LinearSolver> ShapeInnerBuilder<OcatLoss, IdentityLink, OcatVariance, S>
    for OcatInnerBuilder
{
    type Inner = OcatInner<S>;

    fn build(
        &self,
        family: Family<OcatLoss, IdentityLink, OcatVariance>,
        x_design: Array2<f64>,
        y: Array1<f64>,
        prior_weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
        opts: PirlsOpts,
    ) -> Self::Inner {
        OcatInner {
            x_design,
            y,
            prior_weights,
            s_list,
            family,
            opts,
            _solver: PhantomData,
        }
    }
}

/// REML/LAML closed-form envelope score for families with shape parameters.
///
/// θ layout: `[log λ, shape_params...]` where `shape_params` has length
/// `family.n_shape_params()`. The score owns a "base" family that's
/// cloned and updated per probe — propagating to both the Loss and the
/// VarianceFn via `Family::set_shape_params` (architecture-assumptions.md
/// §E2). The `B` type parameter picks the inner solver (PIRLS vs ocat).
///
/// The `P: Profile<L>` type parameter picks the dispersion convention at
/// score-construction time (closes audit §80 — replaces the prior runtime
/// `family.loss.fixed_dispersion().unwrap_or(1.0)` branch):
///
/// - `FixedAtOneProfile` — Ocat, scat/TDist, NegBin (φ ≡ 1).
/// - `OwnedByLossProfile` — Tweedie (φ lives on the Loss, updated per
///   outer probe via `set_shape_params`).
///
/// Gradient strategy: analytic envelope for the `log λ` component (same
/// as Gaussian/Bernoulli), `Loss::analytic_shape_score_gradient` for the
/// shape components when the family supplies it, central FD otherwise.
pub struct ShapeAwareEnvelopeScore<L, K, V, B, P, S = CholeskySolver>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S>,
    P: Profile<L>,
    S: LinearSolver,
{
    pub x_design: Array2<f64>,
    pub y: Array1<f64>,
    pub prior_weights: Option<Array1<f64>>,
    /// Per-term penalty blocks. 94b restricts shape-aware families
    /// (TDist/scat, NegBin, Tweedie, Ocat) to **single-smooth**:
    /// `s_list.len() == 1`. A multi-smooth shape-aware port requires
    /// adding per-term ρ_j to the joint outer-Newton's θ layout
    /// (currently `[ρ, shape_0, …, shape_{n_shape-1}]`) — tracked as a
    /// follow-up.
    pub s_list: Vec<Array2<f64>>,
    /// "Base" family — cloned per probe, then shape params updated from θ.
    pub family_base: Family<L, K, V>,
    /// Per-term rank, must have `len() == 1` for shape-aware families
    /// (see `s_list` docs).
    pub rank_s_list: Vec<usize>,
    pub mp: usize,
    pub log_pseudo_det_s_list: Vec<f64>,
    pub coords: CoordsKind,
    pub pirls_opts: PirlsOpts,
    pub inner_builder: B,
    pub profile: P,
    pub _solver: PhantomData<S>,
}

/// PIRLS-driven shape-aware score — what TDist/scat, NegBin use.
/// (φ fixed at 1 — for Tweedie use `ShapeAwarePirlsScoreOwnedPhi`.)
pub type ShapeAwarePirlsScore<L, K, V> = ShapeAwareEnvelopeScore<
    L,
    K,
    V,
    PirlsInnerBuilder,
    FixedAtOneProfile,
    CholeskySolver,
>;

/// PIRLS-driven shape-aware score with φ read live off the Loss — Tweedie.
pub type ShapeAwarePirlsScoreOwnedPhi<L, K, V> = ShapeAwareEnvelopeScore<
    L,
    K,
    V,
    PirlsInnerBuilder,
    OwnedByLossProfile,
    CholeskySolver,
>;

/// Ocat-driven shape-aware score — what `fit_ocat_cr` uses (φ ≡ 1).
pub type ShapeAwareOcatScore = ShapeAwareEnvelopeScore<
    OcatLoss,
    IdentityLink,
    OcatVariance,
    OcatInnerBuilder,
    FixedAtOneProfile,
    CholeskySolver,
>;

impl<L, K, V, B, P, S> ScoreDerivatives for ShapeAwareEnvelopeScore<L, K, V, B, P, S>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S>,
    P: Profile<L>,
    S: LinearSolver,
{
    fn dim(&self) -> usize {
        // 94b: shape-aware is single-smooth only; theta = [ρ, shape...].
        debug_assert_eq!(
            self.s_list.len(),
            1,
            "ShapeAwareEnvelopeScore restricted to single-smooth (s_list.len() == 1)"
        );
        1 + self.family_base.n_shape_params()
    }

    fn coords(&self) -> CoordsKind {
        self.coords.clone()
    }

    fn value(&self, theta: &Array1<f64>) -> Result<f64> {
        self.compute_value(theta)
    }

    fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
        self.compute_value_grad(theta)
    }

    fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        // Coupled value/grad/Hess at θ. The Hessian path uses ONE inner
        // PIRLS fit at θ (the same one that produces (v, g)) and then
        // does central FD on the analytic gradient with **frozen β̂** at
        // the ±h probes. This is v0.x's `tweedie_theta_grad_hess_analytic`
        // recipe generalised — at converged β̂ the gradient's β-chain is
        // zero by envelope theorem, so FD-on-grad-at-frozen-β̂ is exact
        // to O(h) in the Hessian (same accuracy as FD-on-grad with
        // re-converged β̂, at 1 PIRLS solve instead of 2d+1).
        self.compute_value_grad_hess_analytical(theta)
    }
}

impl<L, K, V, B, P, S> ShapeAwareEnvelopeScore<L, K, V, B, P, S>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S>,
    P: Profile<L>,
    S: LinearSolver,
{
    /// Build a family with the shape params from θ and run the inner solve.
    /// Returns the inner fit plus the rebuilt family (so the score body
    /// can read `loss.saturated_log_lik` / `loss.fixed_dispersion`
    /// consistent with the params).
    fn fit_inner_at(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(GaussianInnerFit<S>, Family<L, K, V>)> {
        let n_shape = self.family_base.n_shape_params();
        debug_assert_eq!(
            theta.len(),
            1 + n_shape,
            "θ has wrong length for shape-aware score"
        );
        let rho = theta[0];
        let shape_slice: Vec<f64> = theta.iter().skip(1).copied().collect();

        let mut family = self.family_base.clone();
        if n_shape > 0 {
            family.set_shape_params(&shape_slice);
        }
        let inner = self.inner_builder.build(
            family.clone(),
            self.x_design.clone(),
            self.y.clone(),
            self.prior_weights.clone(),
            self.s_list.clone(),
            self.pirls_opts.clone(),
        );
        let fit = inner.fit(&Array1::from_vec(vec![rho]))?;
        Ok((fit, family))
    }

    /// Assemble the score value from a converged inner fit at the current θ.
    ///
    /// Extended-family REML/LAML (mgcv `gam.fit5.r:~1003`, v0.x
    /// `src/reml/mod.rs:483`):
    ///   REML = D/(2φ) - Σ ls + log|H|/2 - log|λS|+/2 - Mp/2·log(2πφ)
    ///
    /// For ocat: `OcatLoss::fixed_dispersion() = Some(1.0)` and
    /// `OcatLoss::saturated_log_lik = 0`, so the formula collapses to
    /// `D/2 + log|H|/2 - log|λS|+/2 - Mp/2·log(2π)` — the same formula
    /// `ShapeAwareOcatScore` used pre-unification.
    fn score_value(
        &self,
        fit: &GaussianInnerFit<S>,
        family: &Family<L, K, V>,
        rho: f64,
    ) -> f64 {
        let lambda = rho.exp();
        let s_beta = self.s_list[0].dot(&fit.beta);
        let bsb: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
        let dp = fit.deviance + lambda * bsb;
        // tr(H⁻¹X'WX) for the Profile signature — unused by both
        // FixedAtOneProfile and OwnedByLossProfile in the shape-aware
        // path, so the cheap `p` upper-bound is fine.
        let tr_hinv_xtwx = fit.p as f64;
        let phi = match self
            .profile
            .dispersion(&family.loss, fit, lambda, bsb, tr_hinv_xtwx, self.mp)
        {
            Some(p) => p,
            None => return 1e12,
        };
        let log_det_h = fit.log_det_a();
        let log_det_lambda_s =
            (self.rank_s_list[0] as f64) * rho + self.log_pseudo_det_s_list[0];
        let ls_sum: f64 = self
            .y
            .iter()
            .map(|&yi| family.loss.saturated_log_lik(yi, phi))
            .sum();
        let two_pi = 2.0 * std::f64::consts::PI;
        let mp = self.mp as f64;
        dp / (2.0 * phi)
            - 0.5 * mp * (two_pi * phi).ln()
            + 0.5 * log_det_h
            - 0.5 * log_det_lambda_s
            - ls_sum
    }

    fn compute_value(&self, theta: &Array1<f64>) -> Result<f64> {
        let (fit, family) = self.fit_inner_at(theta)?;
        Ok(self.score_value(&fit, &family, theta[0]))
    }

    fn compute_value_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
        let (fit, family) = self.fit_inner_at(theta)?;
        let rho = theta[0];
        let lambda = rho.exp();
        let v = self.score_value(&fit, &family, rho);

        // Analytic envelope gradient component for log λ.
        let s_beta = self.s_list[0].dot(&fit.beta);
        let bsb: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
        let tr_hinv_s = fit.trace_a_inv(self.s_list[0].view());
        let tr_hinv_xtwx = fit.p as f64;
        let phi = self
            .profile
            .dispersion(&family.loss, &fit, lambda, bsb, tr_hinv_xtwx, self.mp)
            .unwrap_or(1.0);
        // ∂REML/∂(log λ) = λβ'Sβ/(2φ) + λ·tr(H⁻¹S)/2 - rank_s/2
        let g_rho = lambda * bsb / (2.0 * phi)
            + 0.5 * lambda * tr_hinv_s
            - 0.5 * (self.rank_s_list[0] as f64);

        // Shape-param gradient: try analytic first; fall back to central FD.
        let n_shape = family.n_shape_params();
        let mut g = Array1::<f64>::zeros(1 + n_shape);
        g[0] = g_rho;

        if n_shape > 0 {
            let n_minus_mp = (fit.n as f64) - (self.mp as f64);
            let dp = fit.deviance + lambda * bsb;
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
                    g[1 + k] = analytic[k];
                }
            } else {
                // FD fallback (Phase-2 default — TDist, NegBin, ocat).
                let h = 1.0e-5;
                for k in 0..n_shape {
                    let mut t_plus = theta.clone();
                    let mut t_minus = theta.clone();
                    t_plus[1 + k] += h;
                    t_minus[1 + k] -= h;
                    let v_plus = self.compute_value(&t_plus)?;
                    let v_minus = self.compute_value(&t_minus)?;
                    g[1 + k] = (v_plus - v_minus) / (2.0 * h);
                }
            }
        }
        Ok((v, g))
    }

    /// Coupled `(value, grad, hess)` — replaces v0.1's
    /// `hess_via_fd_on_grad` (2d full PIRLS solves per outer Newton iter).
    ///
    /// Recipe (per v0.x `src/reml/tweedie_joint.rs::
    /// tweedie_theta_grad_hess_analytic`): ONE PIRLS solve at θ_center
    /// → value + gradient. Hessian via partial-freeze central FD on the
    /// analytic gradient — log-λ row re-converges PIRLS (β-chain matters
    /// for λ); shape rows freeze β̂ (envelope theorem). For families
    /// without `analytic_shape_score_gradient`, falls back to the v0.1
    /// full FD-on-grad path. Type-level dispatch via the trait method's
    /// `Some(...)` / `None` — no string config.
    ///
    /// `Loss::analytic_shape_score_hessian` is an optional override for
    /// the shape×shape block (defaults `None`; currently no gammon family
    /// uses it — v0.x's FD-on-analytic-grad converges without it).
    fn compute_value_grad_hess_analytical(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let (fit, family) = self.fit_inner_at(theta)?;
        let rho = theta[0];
        let value = self.score_value(&fit, &family, rho);
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

        let mut hess = if has_analytic_shape_grad {
            self.hess_via_fd_frozen_beta(theta, &fit, &ctx)?
        } else {
            self.hess_via_fd_on_grad(theta)?
        };

        // Optional family-supplied closed-form shape×shape block.
        // Currently `None` for every gammon family — hook for future ports.
        if n_shape > 0 {
            let lambda = rho.exp();
            let s_beta = self.s_list[0].dot(&fit.beta);
            let bsb: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            let dp = fit.deviance + lambda * bsb;
            let n_minus_mp = (fit.n as f64) - (self.mp as f64);
            if let Some(block) = family.loss.analytic_shape_score_hessian(
                self.y.view(),
                fit.mu.view(),
                dp,
                n_minus_mp,
                ctx.phi_center,
            ) {
                debug_assert_eq!(block.shape(), &[n_shape, n_shape]);
                for j in 0..n_shape {
                    for k in 0..n_shape {
                        hess[[1 + j, 1 + k]] = block[[j, k]];
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
        let mut hess = Array2::<f64>::zeros((d, d));
        // Re-converge for log-λ direction (i == 0).
        let eps_rho = 1.0e-4;
        let mut t_plus = theta.clone();
        let mut t_minus = theta.clone();
        t_plus[0] += eps_rho;
        t_minus[0] -= eps_rho;
        let (_, g_plus_rho) = self.compute_value_grad(&t_plus)?;
        let (_, g_minus_rho) = self.compute_value_grad(&t_minus)?;
        for j in 0..d {
            hess[[j, 0]] = (g_plus_rho[j] - g_minus_rho[j]) / (2.0 * eps_rho);
        }
        // Frozen-β̂ for shape directions (i ≥ 1).
        let eps_shape = 1.0e-5;
        for i in 1..d {
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

    /// v0.1 fallback path — central FD on the gradient with FULL PIRLS
    /// re-converge at each ±h probe. Used by families without an
    /// analytic shape gradient (TDist, NegBin, Ocat in gammon) where the
    /// frozen-β̂ Hessian is structurally inconsistent with the
    /// FD-on-value gradient.
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

    /// Gradient at θ given an already-converged inner fit at θ. Mirrors
    /// `compute_value_grad`'s gradient block (no second PIRLS solve).
    /// Returns the gradient plus a `FrozenBetaCtx` that the FD probes
    /// reuse to skip the per-probe inner solve.
    fn eval_grad_with_fit(
        &self,
        theta: &Array1<f64>,
        fit: &GaussianInnerFit<S>,
        family: &Family<L, K, V>,
    ) -> Result<(Array1<f64>, FrozenBetaCtx)> {
        let rho = theta[0];
        let lambda = rho.exp();
        let s_beta = self.s_list[0].dot(&fit.beta);
        let bsb: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
        let tr_hinv_s = fit.trace_a_inv(self.s_list[0].view());
        let tr_hinv_xtwx = fit.p as f64;
        let phi_center = self
            .profile
            .dispersion(&family.loss, fit, lambda, bsb, tr_hinv_xtwx, self.mp)
            .unwrap_or(1.0);

        let g_rho = lambda * bsb / (2.0 * phi_center)
            + 0.5 * lambda * tr_hinv_s
            - 0.5 * (self.rank_s_list[0] as f64);

        let n_shape = family.n_shape_params();
        let mut g = Array1::<f64>::zeros(1 + n_shape);
        g[0] = g_rho;
        if n_shape > 0 {
            let n_minus_mp = (fit.n as f64) - (self.mp as f64);
            let dp = fit.deviance + lambda * bsb;
            if let Some(analytic) = family.loss.analytic_shape_score_gradient(
                self.y.view(),
                fit.mu.view(),
                dp,
                n_minus_mp,
                phi_center,
            ) {
                debug_assert_eq!(analytic.len(), n_shape);
                for k in 0..n_shape {
                    g[1 + k] = analytic[k];
                }
            } else {
                // FD fallback (TDist, NegBin, ocat): runs PIRLS at θ ± h.
                // Cost-wise this matches the old path for the gradient eval
                // (those families don't have analytic gradients to speed
                // up); the Hessian still wins via frozen-β̂ FD.
                let h = 1.0e-5;
                for k in 0..n_shape {
                    let mut t_plus = theta.clone();
                    let mut t_minus = theta.clone();
                    t_plus[1 + k] += h;
                    t_minus[1 + k] -= h;
                    let v_plus = self.compute_value(&t_plus)?;
                    let v_minus = self.compute_value(&t_minus)?;
                    g[1 + k] = (v_plus - v_minus) / (2.0 * h);
                }
            }
        }
        Ok((
            g,
            FrozenBetaCtx {
                bsb,
                tr_hinv_s,
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
    fn eval_grad_frozen_beta(
        &self,
        theta: &Array1<f64>,
        fit: &GaussianInnerFit<S>,
        ctx: &FrozenBetaCtx,
    ) -> Result<Array1<f64>> {
        let n_shape = self.family_base.n_shape_params();
        debug_assert_eq!(theta.len(), 1 + n_shape);
        let rho = theta[0];
        let lambda = rho.exp();
        let shape_slice: Vec<f64> = theta.iter().skip(1).copied().collect();

        let mut family = self.family_base.clone();
        if n_shape > 0 {
            family.set_shape_params(&shape_slice);
        }

        // φ at the perturbed family — OwnedByLossProfile (Tweedie) reads
        // `loss.phi`; FixedAtOneProfile stays at 1. Either way the
        // frozen-fit handles (bsb, tr_hinv_s) are reused.
        let phi = self
            .profile
            .dispersion(&family.loss, fit, lambda, ctx.bsb, fit.p as f64, self.mp)
            .unwrap_or(ctx.phi_center);

        let g_rho = lambda * ctx.bsb / (2.0 * phi)
            + 0.5 * lambda * ctx.tr_hinv_s
            - 0.5 * (self.rank_s_list[0] as f64);

        let mut g = Array1::<f64>::zeros(1 + n_shape);
        g[0] = g_rho;
        if n_shape > 0 {
            let dp = ctx.deviance + lambda * ctx.bsb;
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
                g[1 + k] = analytic[k];
            }
        }
        Ok(g)
    }
}
