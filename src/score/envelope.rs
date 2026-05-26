//! `EnvelopeScore` — 1-D outer Newton REML/LAML closed-form score,
//! no shape parameters.
//!
//! Important identity used throughout: since `A := H = X'WX + λS`,
//!
//! ```text
//!   tr(H⁻¹ X'WX) = tr(H⁻¹ (A - λS)) = p - λ·tr(H⁻¹ S)
//! ```
//!
//! So we never need to form `X'WX` separately — one `tr(H⁻¹ S)` solve
//! gives both quantities. This is what lets PIRLS-iterative families
//! (where W changes each iteration) share the Gaussian score body
//! without re-allocating.
//!
//! `S: LinearSolver` (default `CholeskySolver`) flows through the inner
//! fit — the score body calls `inner_fit.log_det_a()` and
//! `inner_fit.trace_a_inv(...)` rather than touching the factor directly.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::error::Result;
use crate::family::Gaussian;
use crate::inner::{CholeskySolver, GaussianClosedFormInner, GaussianInnerFit, LinearSolver};
use crate::traits::{CoordsKind, InnerSolver, Loss, ScoreDerivatives};

use super::profile::{MgcvTwoSigmaProfile, Profile};

/// Closed-form REML / LAML envelope score, generic over the loss `L`,
/// the inner solver `I`, the dispersion profile `P`, and the linear
/// backend `S`. `K` and `V` are erased — the score only needs `Loss`
/// (for `saturated_log_lik`) and the converged inner fit's
/// β / working_weights / deviance. `P` names the σ² convention (see
/// `Profile` trait above).
///
/// This is the trait architecture's keystone: a SINGLE score impl serves
/// every family that fits in the `RemlScoreParts` mould. Plan §3.4 calls
/// this the `ClosedFormEnvelope<F, B>` impl; we name it `EnvelopeScore`.
pub struct EnvelopeScore<
    L: Loss,
    I: InnerSolver<Fit = GaussianInnerFit<S>>,
    P: Profile<L>,
    S: LinearSolver = CholeskySolver,
> {
    pub inner: I,
    pub loss: L,
    pub profile: P,
    /// Per-term penalty list `Vec<S_j>` in current basis coords. Single-
    /// smooth fits use `s_list.len() == 1`; multi-smooth `Additive` fits
    /// use one block per smoothing parameter.
    pub s_list: Vec<Array2<f64>>,
    /// Per-term ranks. The score's `log|λS|+` term is
    /// `Σ_j (rank_j · ρ_j + log_pseudo_det_j)`.
    pub rank_s_list: Vec<usize>,
    pub mp: usize,
    /// Per-term `log|S_j|+` at `λ_j = 1`.
    pub log_pseudo_det_s_list: Vec<f64>,
    pub coords: CoordsKind,
    /// Raw response `y` — needed for `Σ ls(y_i)` in the score formula.
    /// (`inner.y` is also available but we keep our own copy so the score
    /// doesn't have to peek inside the InnerSolver.)
    pub y: Array1<f64>,
    pub _solver: PhantomData<S>,
}

/// Phase-0 / Phase-1 convenience type alias for the Gaussian one-Cholesky
/// inner with the mgcv two-σ² convention. PIRLS-iterative families wire
/// `EnvelopeScore<L, PirlsInner<L, K, V>, P>` directly.
pub type GaussianClosedFormScore =
    EnvelopeScore<Gaussian, GaussianClosedFormInner<CholeskySolver>, MgcvTwoSigmaProfile, CholeskySolver>;

impl<L, I, P, S> EnvelopeScore<L, I, P, S>
where
    L: Loss,
    I: InnerSolver<Fit = GaussianInnerFit<S>>,
    P: Profile<L>,
    S: LinearSolver,
{
    /// Generic constructor — accepts any `(Loss, InnerSolver, Profile)`
    /// triple. Used by PIRLS-based fit entry points (`fit_binomial_cr`,
    /// `fit_poisson_cr`, …) for a consistent build pattern across families.
    pub fn with_inner(
        inner: I,
        loss: L,
        profile: P,
        y: Array1<f64>,
        s_list: Vec<Array2<f64>>,
        rank_s_list: Vec<usize>,
        mp: usize,
        log_pseudo_det_s_list: Vec<f64>,
    ) -> Self {
        debug_assert_eq!(s_list.len(), rank_s_list.len());
        debug_assert_eq!(s_list.len(), log_pseudo_det_s_list.len());
        Self {
            inner,
            loss,
            profile,
            s_list,
            rank_s_list,
            mp,
            log_pseudo_det_s_list,
            coords: CoordsKind::Identity,
            y,
            _solver: PhantomData,
        }
    }
}

impl GaussianClosedFormScore {
    /// Phase-0 convenience constructor — wires the closed-form Gaussian
    /// inner with the mgcv two-σ² profile. Equivalent to
    /// `with_inner(GaussianClosedFormInner::new(...), Gaussian, MgcvTwoSigmaProfile, ...)`.
    pub fn new(
        x_design: Array2<f64>,
        y: Array1<f64>,
        s_list: Vec<Array2<f64>>,
        weights: Option<Array1<f64>>,
        rank_s_list: Vec<usize>,
        mp: usize,
        log_pseudo_det_s_list: Vec<f64>,
    ) -> Self {
        let inner = GaussianClosedFormInner::<CholeskySolver>::new(
            x_design,
            y.clone(),
            weights,
            s_list.clone(),
        );
        Self::with_inner(
            inner,
            Gaussian,
            MgcvTwoSigmaProfile,
            y,
            s_list,
            rank_s_list,
            mp,
            log_pseudo_det_s_list,
        )
    }
}

impl<L, I, P, S> ScoreDerivatives for EnvelopeScore<L, I, P, S>
where
    L: Loss,
    I: InnerSolver<Fit = GaussianInnerFit<S>>,
    P: Profile<L>,
    S: LinearSolver,
{
    fn dim(&self) -> usize {
        self.s_list.len()
    }

    fn coords(&self) -> CoordsKind {
        self.coords.clone()
    }

    fn value(&self, theta: &Array1<f64>) -> Result<f64> {
        let (v, _, _) = self.value_grad_hess(theta)?;
        Ok(v)
    }

    fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
        let (v, g, _) = self.value_grad_hess(theta)?;
        Ok((v, g))
    }

    fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let (v, g) = self.compute_value_grad(theta)?;
        // Phase 1 Hessian: central FD on the gradient. The FD lives on the
        // score (not the outer loop) so the structural defence against
        // closed-form-vs-FD drift holds — `outer.rs` only sees
        // `value_grad_hess`.
        let hess = self.hess_via_fd(theta)?;
        Ok((v, g, hess))
    }
}

impl<L, I, P, S> EnvelopeScore<L, I, P, S>
where
    L: Loss,
    I: InnerSolver<Fit = GaussianInnerFit<S>>,
    P: Profile<L>,
    S: LinearSolver,
{
    /// Coupled `(value, grad)` — the actual closed-form work. `value_grad_hess`
    /// is a thin wrapper that adds an FD Hessian.
    fn compute_value_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
        debug_assert_eq!(
            theta.len(),
            self.s_list.len(),
            "EnvelopeScore: theta length {} must equal s_list length {}",
            theta.len(),
            self.s_list.len()
        );
        let n_terms = self.s_list.len();
        // λ_j = exp(ρ_j) per term.
        let lambda_j: Vec<f64> = theta.iter().map(|&r| r.exp()).collect();
        let inner: GaussianInnerFit<S> = self.inner.fit(theta)?;

        // Per-term β' S_j β and tr(H⁻¹ S_j). The combined `λ S β` becomes
        // `Σ_j λ_j (S_j β)`; the score uses these per-term to compute the
        // multi-d gradient `∂REML/∂ρ_j = λ_j β'S_j β/(2σ²) + λ_j tr(H⁻¹ S_j)/2
        // - rank_j/2`. v0.x's `reml_gradient_multi_*` follows the same shape.
        let mut bsb_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut tr_hinv_s_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut bsb_total = 0.0_f64; // Σ_j λ_j β'S_jβ — used in dp / σ²-eq
        let mut tr_hinv_s_lambda_total = 0.0_f64; // Σ_j λ_j tr(H⁻¹S_j)
        for j in 0..n_terms {
            let s_j = &self.s_list[j];
            let s_beta = s_j.dot(&inner.beta);
            let bsb_j: f64 = inner.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            let tr_hinv_s_j = inner.trace_a_inv(s_j.view());
            bsb_per_term.push(bsb_j);
            tr_hinv_s_per_term.push(tr_hinv_s_j);
            bsb_total += lambda_j[j] * bsb_j;
            tr_hinv_s_lambda_total += lambda_j[j] * tr_hinv_s_j;
        }
        // For the Tk·KK' Newton path (non-canonical-link, single-smooth
        // only — see PirlsInner): the cached `tr_a_newton_inv_s` is the
        // Newton-A trace against the combined `s_total`. Use it for the
        // gradient's `λ·tr(H⁻¹S)/2` aggregate (mirrors v0.x); per-term
        // Newton traces are not yet wired (multi-smooth Newton-IRLS is
        // out of scope for 94b).
        let tr_hinv_s_combined_for_grad = inner
            .tk_kkt_inputs
            .as_ref()
            .map(|tk| tk.tr_a_newton_inv_s)
            .unwrap_or(tr_hinv_s_lambda_total);

        // Identity: tr(H⁻¹ X'WX) = p - Σ_j λ_j·tr(H⁻¹ S_j). Fisher
        // version for the σ²-grad denominator (mgcv convention).
        let tr_hinv_xtwx = (inner.p as f64) - tr_hinv_s_lambda_total;

        // `log|H|` — defaults to the backend's `log|A|` off the Fisher
        // factor; PIRLS overrides this with the Newton-W `Σ log|λ_i|`
        // when the loss opts into the Newton path (non-canonical
        // InverseGaussian + log) so we match v0.x
        // `src/reml/mod.rs:436-459`. `tr(H⁻¹S)` continues to use Fisher
        // H per v0.x's `system.tr_a` convention (the score's
        // `(n − tr_a)` denominator expects Fisher).
        let log_det_h_fisher = inner.log_det_a();
        let log_det_h = inner.log_det_h_override.unwrap_or(log_det_h_fisher);
        // log|λS|+ for multi-smooth: Σ_j (rank_j · ρ_j + log_pseudo_det_j).
        let mut log_det_lambda_s = 0.0_f64;
        for j in 0..n_terms {
            log_det_lambda_s +=
                (self.rank_s_list[j] as f64) * theta[j] + self.log_pseudo_det_s_list[j];
        }

        // σ² dispatch via the `Profile` type parameter — type-level choice,
        // not a runtime branch (architecture-assumptions.md §D3).
        //
        // Multi-smooth lift: the Profile signature takes a scalar `lambda`
        // and scalar `bsb`; here we pass `lambda = 1` and the pre-computed
        // `bsb_total = Σ_j λ_j β'S_jβ` so `dp = D + 1·bsb_total` matches
        // the multi-smooth Dp.
        let score_sigma2 = match self.profile.dispersion(
            &self.loss,
            &inner,
            1.0,
            bsb_total,
            tr_hinv_xtwx,
            self.mp,
        ) {
            Some(phi) => phi,
            None => return Ok((1e12, Array1::zeros(n_terms))),
        };

        // `Dp = D + Σ_j λ_j·β'S_jβ`.
        let dp = inner.deviance + bsb_total;

        // Σ ls(y_i; σ²) — saturated log-likelihood sum at the score's σ².
        // Gamma/InverseGaussian's sat_lik depend on σ² via the φ-term that
        // Phase-2b v0.2 port reinstated; other families ignore the scale arg.
        let ls_sum: f64 = self
            .y
            .iter()
            .map(|&yi| self.loss.saturated_log_lik(yi, score_sigma2))
            .sum();

        // GamFit3 form (mgcv `gam.fit3.r:616-617`, v0.x
        // `src/pirls/mod.rs::ScoreFormula::GamFit3`):
        //
        //   REML = Dp/(2σ²) - Mp/2·log(2πσ²) + log|H|/2 - log|λS|+/2 - Σ ls
        //
        // Phase-2b v0.2 port (2026-05-24): switched from the Gaussian-
        // shortcut `+(n-Mp)/2·log(2πσ²) - Σls=0` form to the general form.
        // The two coincide for σ²-independent sat_lik (Gaussian's
        // `-n/2·log(2πφ)` exactly fills the `+n/2·log(2πφ)` gap between
        // `(n-Mp)/2` and `-Mp/2`); for σ²-DEPENDENT sat_lik (Gamma,
        // InverseGaussian) the general form is the only correct one.
        let two_pi = 2.0 * std::f64::consts::PI;
        let mp_f = self.mp as f64;
        let reml = dp / (2.0 * score_sigma2)
            - 0.5 * mp_f * (two_pi * score_sigma2).ln()
            + 0.5 * log_det_h
            - 0.5 * log_det_lambda_s
            - ls_sum;

        // Per-term gradient: ∂REML/∂ρ_j = λ_j β'S_jβ/(2σ²) + λ_j tr(H⁻¹S_j)/2
        //   - rank_j/2  (mgcv `reml_gradient_multi_*` shape).
        let mut g = Array1::<f64>::zeros(n_terms);
        for j in 0..n_terms {
            g[j] = lambda_j[j] * bsb_per_term[j] / (2.0 * score_sigma2)
                + 0.5 * lambda_j[j] * tr_hinv_s_per_term[j]
                - 0.5 * (self.rank_s_list[j] as f64);
        }

        // Tk·KK' contribution for non-canonical-link families. The W
        // matrix used in `log|H|` depends on β (through μ), so
        // `d(log|H|)/dρ` carries the explicit term
        // `Σᵢ a1[i] · η₁[i] · sign(w[i]) · lev_uw[i]` (v0.x
        // `src/reml/mod.rs::reml_gradient_mgcv_exact_ift_inner_at_beta`).
        // Currently wired only for single-smooth (s_list.len() == 1) —
        // multi-smooth Tk·KK' would need per-term η₁_j = -λ_j·X·A⁻¹·S_j·β
        // derivatives. Falls back to the Fisher-trace gradient when
        // multi-smooth + use_newton_irls is requested (debug_assert
        // guards in PirlsInner ensure this).
        if let Some(ref tk) = inner.tk_kkt_inputs {
            debug_assert_eq!(
                n_terms, 1,
                "EnvelopeScore: Tk·KK' Newton-IRLS path only supports single-smooth fits"
            );
            let n = inner.n;
            let mut tk_kkt = 0.0_f64;
            for i in 0..n {
                tk_kkt += tk.a1[i] * tk.eta1[i] * tk.lev_uw[i];
            }
            // Single-smooth: also rewrite the trace contribution to use
            // the Newton-A version so the Tk·KK' machinery is self-
            // consistent (matches v0.x `reml_gradient_mgcv_exact_ift_newton_at_beta`).
            g[0] = lambda_j[0] * bsb_per_term[0] / (2.0 * score_sigma2)
                + 0.5 * lambda_j[0] * tr_hinv_s_combined_for_grad
                - 0.5 * (self.rank_s_list[0] as f64)
                + 0.5 * tk_kkt;
        }

        Ok((reml, g))
    }

    fn hess_via_fd(&self, theta: &Array1<f64>) -> Result<Array2<f64>> {
        let d = theta.len();
        let mut h = Array2::<f64>::zeros((d, d));
        let eps = 1.0e-5;
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
