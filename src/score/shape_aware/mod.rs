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
//!
//! Submodule layout (post-refactor wave 5 — file size hygiene, >700 LOC):
//! - `mod.rs` (this file) — public struct, builder trait + impls,
//!   `ScoreDerivatives` entry points, `FrozenBetaCtx`.
//! - `score_value` — `fit_inner_at`, `score_value`, `compute_value`.
//! - `gradient` — `compute_value_grad`, `compute_rho_envelope_gradient`,
//!   `eval_grad_with_fit`, `eval_grad_frozen_beta`,
//!   `analytic_shape_grad_via_ift`.
//! - `hessian` — `compute_value_grad_hess_analytical`,
//!   `hess_via_fd_frozen_beta`, `hess_via_fd_on_value`,
//!   `hess_via_fd_on_grad`.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::family::{Family, IdentityLink, OcatLoss, OcatVariance};
use crate::inner::{
    CholeskySolver, GaussianInnerFit, LinearSolver, OcatInner, PirlsInner, PirlsOpts,
};
use crate::traits::{CoordsKind, InnerSolver, Link, Loss, ScoreDerivatives, VarianceFn};

use super::profile::{FixedAtOneProfile, OwnedByLossProfile, Profile};

mod gradient;
mod hessian;
mod score_value;

/// Frozen-β̂ context shared between `eval_grad_with_fit` and
/// `eval_grad_frozen_beta`. Holds the converged-inner quantities that
/// stay constant across ±h shape probes.
///
/// Per-term vectors (`bsb_per_term`, `tr_hinv_s_per_term`, length T) feed
/// the per-ρ_j envelope gradient at frozen β̂; `bsb_total = Σ_j λ_j · bsb_j`
/// feeds the φ formula / shape gradient.
#[derive(Clone)]
pub(super) struct FrozenBetaCtx {
    pub(super) bsb_per_term: Vec<f64>,
    pub(super) tr_hinv_s_per_term: Vec<f64>,
    pub(super) bsb_total: f64,
    pub(super) phi_center: f64,
    pub(super) n_minus_mp: f64,
    pub(super) deviance: f64,
}

/// Build an `InnerSolver` from a freshly shape-synced family + the score's
/// owned design fields. The score body uses this to rebuild the inner per
/// outer probe (shape params shift every probe; the inner must see the
/// current family).
///
/// Two concrete impls in gamrs:
/// - `PirlsInnerBuilder<S>` (generic over `L, K, V, S`) — drives TDist/scat,
///   Tweedie, NegBin via the standard PIRLS loop.
/// - `OcatInnerBuilder<S>` — drives the ocat extended family via `OcatInner`,
///   constrained to `Family<OcatLoss, IdentityLink, OcatVariance>` (the
///   only valid Loss/Link/Variance triple for ordered categorical).
pub trait ShapeInnerBuilder<
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver = CholeskySolver,
>
{
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
