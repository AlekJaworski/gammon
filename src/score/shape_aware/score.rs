//! `ShapeAwareEnvelopeScore` struct, type aliases, the `ScoreDerivatives`
//! trait impl, plus the orchestrating `fit_inner_at` / `score_value`
//! helpers used by both the gradient and Hessian paths.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::family::{Family, IdentityLink, OcatLoss, OcatVariance};
use crate::inner::{CholeskySolver, GaussianInnerFit, LinearSolver, PirlsOpts};
use crate::traits::{CoordsKind, InnerSolver, Link, Loss, ScoreDerivatives, VarianceFn};

use super::super::profile::{FixedAtOneProfile, OwnedByLossProfile, Profile};
use super::builder::{OcatInnerBuilder, PirlsInnerBuilder, ShapeInnerBuilder};

/// Frozen-β̂ context shared between `eval_grad_with_fit` and
/// `eval_grad_frozen_beta`. Holds the converged-inner quantities that
/// stay constant across ±h shape probes.
///
/// Per-term vectors (`bsb_per_term`, `tr_hinv_s_per_term`, length T) feed
/// the per-ρ_j envelope gradient at frozen β̂; `bsb_total = Σ_j λ_j · bsb_j`
/// feeds the φ formula / shape gradient (where the family sees the
/// penalty contribution aggregated). `deviance`, `phi_center`,
/// `n_minus_mp` are scalars.
#[derive(Clone)]
pub(super) struct FrozenBetaCtx {
    pub(super) bsb_per_term: Vec<f64>,
    pub(super) tr_hinv_s_per_term: Vec<f64>,
    pub(super) bsb_total: f64,
    pub(super) phi_center: f64,
    pub(super) n_minus_mp: f64,
    pub(super) deviance: f64,
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
    /// Per-term penalty blocks. Multi-smooth in 0.2 — joint outer-Newton
    /// optimises `θ = [ρ₁, …, ρ_T, shape_0, …, shape_{n_shape-1}]` where
    /// `T = s_list.len()`. The shape-aware drivers (TDist/scat, NegBin,
    /// Tweedie, Ocat) all flow through this score type unchanged.
    pub s_list: Vec<Array2<f64>>,
    /// "Base" family — cloned per probe, then shape params updated from θ.
    pub family_base: Family<L, K, V>,
    /// Per-term rank, length `T = s_list.len()`.
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
pub type ShapeAwarePirlsScore<L, K, V> =
    ShapeAwareEnvelopeScore<L, K, V, PirlsInnerBuilder, FixedAtOneProfile, CholeskySolver>;

/// PIRLS-driven shape-aware score with φ read live off the Loss — Tweedie.
pub type ShapeAwarePirlsScoreOwnedPhi<L, K, V> =
    ShapeAwareEnvelopeScore<L, K, V, PirlsInnerBuilder, OwnedByLossProfile, CholeskySolver>;

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
        // 0.2: multi-smooth — θ = [ρ_1, …, ρ_T, shape_0, …, shape_{n_shape-1}].
        self.s_list.len() + self.family_base.n_shape_params()
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

    fn axis_step_caps(&self) -> Option<Vec<f64>> {
        // ρ axes: 5.0 per axis (matches gamrs's global default, mgcv's
        // unspecified-default Newton step magnitude in log-λ space).
        // Shape axes: family-supplied (e.g. ocat θ → 0.5, Tweedie p_t → 2.0).
        let n_terms = self.s_list.len();
        let mut caps: Vec<f64> = vec![5.0; n_terms];
        caps.extend(self.family_base.loss.shape_axis_step_caps());
        Some(caps)
    }

    fn axis_bounds(&self) -> Option<Vec<(f64, f64)>> {
        // ρ axes: effectively unbounded (large range — log λ saturation
        // is governed by the gradient flattening, not a hard cap). Shape
        // axes: family-supplied (ocat θ ∈ [-10, 10] etc.).
        let n_terms = self.s_list.len();
        let mut bnds: Vec<(f64, f64)> = vec![(-50.0, 50.0); n_terms];
        bnds.extend(self.family_base.loss.shape_axis_bounds());
        Some(bnds)
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
    pub(super) fn fit_inner_at(
        &self,
        theta: &Array1<f64>,
    ) -> Result<(GaussianInnerFit<S>, Family<L, K, V>)> {
        let n_terms = self.s_list.len();
        let n_shape = self.family_base.n_shape_params();
        debug_assert_eq!(
            theta.len(),
            n_terms + n_shape,
            "θ has wrong length for shape-aware score"
        );
        let rho_slice: Array1<f64> = theta.slice(ndarray::s![..n_terms]).to_owned();
        let shape_slice: Vec<f64> = theta.iter().skip(n_terms).copied().collect();

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
        let fit = inner.fit(&rho_slice)?;
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
    pub(super) fn score_value(
        &self,
        fit: &GaussianInnerFit<S>,
        family: &Family<L, K, V>,
        rho_slice: &[f64],
    ) -> f64 {
        let n_terms = self.s_list.len();
        debug_assert_eq!(rho_slice.len(), n_terms);

        // Per-family mgcv-style rank adjustment (ocat: −1; others: 0).
        let rank_adj = family.loss.score_rank_adjustment();
        // Per-term bsb_j = β'S_j β + aggregate via λ_j. dp = D + Σ_j λ_j β'S_jβ.
        let mut bsb_total = 0.0_f64;
        let mut log_det_lambda_s = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = self.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            let lambda_j = rho_slice[j].exp();
            bsb_total += lambda_j * bsb_j;
            let adj_rank_j = ((self.rank_s_list[j] as i32 + rank_adj).max(1)) as f64;
            log_det_lambda_s +=
                adj_rank_j * rho_slice[j] + self.log_pseudo_det_s_list[j];
        }
        let dp = fit.deviance + bsb_total;
        // tr(H⁻¹X'WX) for the Profile signature — unused by both
        // FixedAtOneProfile and OwnedByLossProfile in the shape-aware
        // path, so the cheap `p` upper-bound is fine.
        let tr_hinv_xtwx = fit.p as f64;
        // Profile sees aggregated `bsb_total` with `lambda = 1` (matches
        // EnvelopeScore convention — envelope.rs:266-273).
        let phi =
            match self
                .profile
                .dispersion(&family.loss, fit, 1.0, bsb_total, tr_hinv_xtwx, self.mp)
            {
                Some(p) => p,
                None => return 1e12,
            };
        // Use the Newton-W `log|H|` override when the inner solver provided
        // one (NegBin / InverseGaussian / similar non-canonical-link
        // families opting into `Loss::use_newton_irls()`). Otherwise the
        // Fisher A factor's `log|H|` is the correct REML object — same
        // selection rule that `EnvelopeScore::evaluate` uses
        // (`src/score/envelope.rs:292`). Without this, the shape-aware
        // path FD-of-value differentiates the Fisher `log|H|` while the
        // analytic shape gradient at `gradient.rs:307-468` uses Newton-A⁻¹
        // in its IFT formula — the resulting mismatch shows up as the
        // 6-23% rel-err on the `log θ` axis of
        // `negbin_multismooth_analytic_grad_matches_fd`.
        let log_det_h = fit.log_det_h_override.unwrap_or_else(|| fit.log_det_a());
        let ls_sum: f64 = self
            .y
            .iter()
            .map(|&yi| family.loss.saturated_log_lik(yi, phi))
            .sum();
        let two_pi = 2.0 * std::f64::consts::PI;
        let mp = self.mp as f64;
        dp / (2.0 * phi) - 0.5 * mp * (two_pi * phi).ln() + 0.5 * log_det_h
            - 0.5 * log_det_lambda_s
            - ls_sum
    }
}
