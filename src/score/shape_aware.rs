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
/// stay constant across ±h shape probes.
///
/// Per-term vectors (`bsb_per_term`, `tr_hinv_s_per_term`, length T) feed
/// the per-ρ_j envelope gradient at frozen β̂; `bsb_total = Σ_j λ_j · bsb_j`
/// feeds the φ formula / shape gradient (where the family sees the
/// penalty contribution aggregated). `deviance`, `phi_center`,
/// `n_minus_mp` are scalars.
#[derive(Clone)]
struct FrozenBetaCtx {
    bsb_per_term: Vec<f64>,
    tr_hinv_s_per_term: Vec<f64>,
    bsb_total: f64,
    phi_center: f64,
    n_minus_mp: f64,
    deviance: f64,
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

    /// Build a family with the shape params from θ and run the inner solve.
    /// Returns the inner fit plus the rebuilt family (so the score body
    /// can read `loss.saturated_log_lik` / `loss.fixed_dispersion`
    /// consistent with the params).
    fn fit_inner_at(&self, theta: &Array1<f64>) -> Result<(GaussianInnerFit<S>, Family<L, K, V>)> {
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
    fn score_value(
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
        let log_det_h = fit.log_det_a();
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

    fn compute_value(&self, theta: &Array1<f64>) -> Result<f64> {
        let n_terms = self.s_list.len();
        let (fit, family) = self.fit_inner_at(theta)?;
        let rho_slice = theta.slice(ndarray::s![..n_terms]).to_vec();
        Ok(self.score_value(&fit, &family, &rho_slice))
    }

    fn compute_value_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
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
                    self.analytic_shape_grad_via_ift(&fit, &level1, n_terms)?;
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
        level1: &crate::traits::Level1ShapeDerivs,
        _n_terms_for_layout: usize,
    ) -> Result<Array1<f64>> {
        let n = fit.n;
        let p = fit.p;
        let n_theta = level1.dth.ncols();
        debug_assert_eq!(level1.dth.nrows(), n);
        debug_assert_eq!(level1.dmuth.shape(), level1.dth.shape());
        debug_assert_eq!(level1.dmu2th.shape(), level1.dth.shape());
        debug_assert_eq!(level1.dmu3.len(), n);

        // dβ/dθ_k = −H⁻¹ · X' · Dmuth[:, k] / 2 via the fit's factor.
        // We solve column-by-column to avoid materialising H⁻¹.
        let mut dbeta_dtheta = Array2::<f64>::zeros((p, n_theta));
        for k in 0..n_theta {
            let dmuth_k = level1.dmuth.column(k);
            let rhs: Array1<f64> = self.x_design.t().dot(&dmuth_k) * 0.5;
            let v = S::solve(&fit.a_factor, rhs.view());
            for r in 0..p {
                dbeta_dtheta[[r, k]] = -v[r];
            }
        }

        // h_diag[i] = X_i' H⁻¹ X_i.  v0.x materialises H⁻¹ once and does
        // O(np²) per row; we do the same but via the fit's factor with
        // column solves of H⁻¹·X' (still O(np²) on dense X, p small).
        // Build A_inv·X' by solving column-wise.
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

        let mut grad = Array1::<f64>::zeros(n_theta);
        for k in 0..n_theta {
            // Envelope: Σᵢ Dth[i, k] = ∂(D + P)/∂θ_k (no β-chain at converged β).
            let mut sum_dth_k = 0.0_f64;
            for i in 0..n {
                sum_dth_k += level1.dth[[i, k]];
            }

            // tr(H⁻¹ ∂H/∂θ_k) = Σᵢ ½ (Dmu2th[i,k] + Dmu3[i] · (X·dβ/dθ_k)[i]) · h_diag[i]
            let mut trace_term = 0.0_f64;
            for i in 0..n {
                let mut x_db_i = 0.0_f64;
                for j in 0..p {
                    x_db_i += self.x_design[[i, j]] * dbeta_dtheta[[j, k]];
                }
                let s_ki = 0.5 * (level1.dmu2th[[i, k]] + level1.dmu3[i] * x_db_i);
                trace_term += s_ki * h_diag[i];
            }
            grad[k] = 0.5 * sum_dth_k + 0.5 * trace_term;
        }
        Ok(grad)
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
    /// the shape×shape block (defaults `None`; currently no gamrs family
    /// uses it — v0.x's FD-on-analytic-grad converges without it).
    fn compute_value_grad_hess_analytical(
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

        let mut hess = if has_analytic_shape_grad {
            self.hess_via_fd_frozen_beta(theta, &fit, &ctx)?
        } else {
            // v0.x recipe: direct central FD on the REML score value.
            // Eliminates the FD-of-FD chain noise that was driving
            // gamrs's saturated-λ over-leap on scat/negbin/ocat.
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

    /// v0.1 fallback path — central FD on the gradient with FULL PIRLS
    /// re-converge at each ±h probe. Retained because Tweedie's mixed
    /// shape×ρ Hessian rows use the analytic-grad-frozen-β route instead.
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
    fn eval_grad_frozen_beta(
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
