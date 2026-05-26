//! Core trait skeletons (per v2 plan §3).

use ndarray::{Array1, Array2, ArrayView2};

use crate::Result;

/// Layer 1 — a basis defines what β means: design matrix, derivatives,
/// and penalty matrices in coefficient space.
///
/// `x` is always `(n_obs, n_input_dims)` so tensor products `te(s(x), s(z))`
/// can be added later without changing the trait shape. Phase 0 univariate
/// bases pass `n_input_dims = 1` and ignore the `axis` parameter on `d1`.
pub trait Basis {
    /// Number of basis functions = length of β when fitted on this basis.
    fn dim(&self) -> usize;

    /// Number of input dimensions the basis consumes. Phase 0 always 1.
    fn input_dim(&self) -> usize;

    /// Design matrix rows for evaluation points. `x.shape() = (n_obs,
    /// input_dim())`. Returns `(n_obs, self.dim())`.
    fn evaluate(&self, x: ArrayView2<f64>) -> Array2<f64>;

    /// `∂design/∂x_axis` evaluated at `x`. Same shape as `evaluate`.
    /// `axis` is the input dimension to differentiate against. For a
    /// univariate basis pass `axis = 0`.
    fn d1(&self, x: ArrayView2<f64>, axis: usize) -> Array2<f64>;

    /// One penalty matrix per smoothing parameter slot. CR spline returns
    /// exactly one; tensor products will return one per marginal.
    fn penalties(&self) -> Vec<Array2<f64>>;
}

/// Layer 1.5 — a linear constraint/rotation on top of an inner basis.
/// Default `Basis` impl is provided via blanket impl downstream.
pub trait BasisTransform {
    type Inner: Basis;
    fn inner(&self) -> &Self::Inner;
    /// Constraint matrix `C` of shape `(k_inner, k_self)`. New basis is
    /// `B_self = B_inner · C`, new penalties are `C' · S · C`.
    fn matrix(&self) -> ArrayView2<'_, f64>;
}

/// Layer 2a — `D(y, μ)` per observation, plus first/second derivatives in
/// μ. PIRLS uses `d_loss_dmu` / `d2_loss_dmu` to build working weights and
/// working response; the score body uses `deviance_per_obs` and
/// `saturated_log_lik`.
pub trait Loss {
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64;
    /// Saturated log-likelihood — required for REML/LAML criterion.
    ///
    /// `scale` is the dispersion σ² at the current outer probe (the σ²
    /// the score body uses in its `Dp/(2σ²) + (n-Mp)/2·log(2π·σ²)` term).
    /// For families whose saturated log-lik is genuinely y-only and
    /// scale-free (Gaussian/Bernoulli/Poisson at canonical units) the
    /// argument is ignored. Gamma, InverseGaussian, scat/TDist and
    /// Tweedie all depend on σ² via lgamma / log terms; those impls
    /// MUST read it (v0.2 port, 2026-05-24).
    fn saturated_log_lik(&self, y: f64, scale: f64) -> f64;
    /// `∂D/∂μ` per observation. Used inside the PIRLS working response.
    /// Default returns `2(y - μ)` for the Gaussian squared-error case.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        -2.0 * (y - mu)
    }
    /// `∂²D/∂μ²` per observation. Default `2.0` for Gaussian.
    fn d2_loss_dmu(&self, _y: f64, _mu: f64) -> f64 {
        2.0
    }
    /// Known dispersion if any; `None` means "profile σ² from the fit"
    /// (Gaussian, Gamma); `Some(1.0)` means "σ² is fixed at 1" (Bernoulli,
    /// Poisson). This is what lets `ClosedFormEnvelopeScore` dispatch
    /// between the profiled-σ² and known-σ² REML formulae.
    fn fixed_dispersion(&self) -> Option<f64> {
        None
    }

    /// Profile σ̂²(ρ) that solves `∂REML/∂σ² = 0` at fixed ρ — i.e. the
    /// σ² that the REML score formula plugs in for THIS family's
    /// saturated log-lik. Default is mgcv's Gaussian closed form
    /// `Dp/(n - Mp)`, which is exact for any family whose sat_lik has
    /// the form `-n/2·log(2πφ) + (y-only piece)` (Gaussian, InverseGaussian,
    /// QuasiPoisson, QuasiBinomial). Gamma overrides — its sat_lik
    /// `k(φ) = -lgamma(1/φ) - log(φ)/φ - 1/φ` makes the profile equation
    /// `F(φ) = dp + 2n[ψ(1/φ) + log φ] + Mp·φ = 0` which needs Newton
    /// (see `src/pirls/mod.rs::estimate_phi_mgcv`).
    ///
    /// `phi_init` is a warm-start hint (typically the previous outer
    /// iteration's φ̂).
    fn profile_score_sigma2(&self, dp: f64, _n_obs: usize, n_minus_mp: f64, _phi_init: f64) -> f64 {
        (dp / n_minus_mp.max(1.0)).max(1e-8)
    }

    /// Number of family-shape parameters this loss owns. Zero for
    /// Gaussian/Bernoulli/Poisson. TDist returns 2 (`log σ²`, `log(ν-2)`).
    /// Tweedie returns 2 (`log φ`, `p_transform`). Used by the outer Newton
    /// to know how to size θ.
    fn n_shape_params(&self) -> usize {
        0
    }
    /// Update this loss's shape parameters from a transformed θ-slice.
    /// The slice has length `n_shape_params()`. Default no-op for stateless
    /// losses. Each Loss documents its own transform (typically `log(·)`
    /// for positive scale params, `log(· - lower_bound)` for bounded ones).
    fn set_shape_params(&mut self, _params: &[f64]) {}
    /// Read the current shape parameters back as the transformed θ-slice.
    /// Inverse of `set_shape_params`. Useful for seeding the outer Newton
    /// from a fitted state.
    fn get_shape_params(&self) -> Vec<f64> {
        Vec::new()
    }

    /// Analytic contribution to the score's gradient w.r.t. THIS loss's
    /// shape parameters, evaluated at the current shape values.
    ///
    /// Returns `Some(grad)` of length `n_shape_params()` if the family has
    /// analytical derivatives for ALL its shape-related score terms (i.e.
    /// the caller can use the returned vector directly as the
    /// `g[1 + k]` slice for shape param k); `None` means the caller should
    /// fall back to FD on the score-value (current Phase-2 default).
    ///
    /// Inputs (all evaluated at the converged PIRLS β̂ for the current
    /// (ρ, shape) probe):
    /// - `y` — response.
    /// - `mu` — fitted μ̂.
    /// - `dp` — `D + λβ'Sβ` at converged β̂ (the "penalised deviance").
    /// - `n_minus_mp` — `n − Mp`, the score formula's denominator coefficient.
    /// - `phi_score` — the score's current σ² (`Profile` output, or
    ///   `fixed_dispersion()` for shape-managed dispersion families like
    ///   Tweedie). The impl uses this for the `D/(2φ)` and `log(2πφ)` terms.
    ///
    /// Envelope-theorem assumption: at PIRLS convergence, ∂(β̂)/∂(shape) does
    /// NOT contribute to the score gradient — only the explicit
    /// shape-dependence does. This matches v0.x's `RemlScoreParts` assembly.
    ///
    /// Phase-1 port (2026-05-24): Tweedie overrides this to use
    /// `crate::special::tweedie_series` analytical derivatives directly,
    /// closing the wrong-local-minimum bug from FD on the series.
    fn analytic_shape_score_gradient(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _mu: ndarray::ArrayView1<f64>,
        _dp: f64,
        _n_minus_mp: f64,
        _phi_score: f64,
    ) -> Option<ndarray::Array1<f64>> {
        None
    }

    /// Analytic Hessian of the score w.r.t. THIS loss's shape parameters,
    /// evaluated at the current shape values. Returns `Some(hess)` of shape
    /// `(n_shape, n_shape)` if the family provides closed-form 2nd
    /// derivatives; `None` means the caller falls back to FD on the
    /// analytic gradient (with frozen β̂, the v0.x recipe).
    ///
    /// Inputs match `analytic_shape_score_gradient` plus optional access
    /// to the converged β̂ via the caller — for the cases we currently
    /// port from v0.x (Tweedie via `tweedie_dd_level1` + Wright-series
    /// second derivatives) only the converged μ̂ and the family's own
    /// shape state are needed, so the trait surface stays the same.
    ///
    /// Default `None` keeps families on the FD-on-analytic-gradient path
    /// — that path is already structurally correct (it differentiates the
    /// analytic envelope gradient) and matches v0.x's
    /// `tweedie_theta_grad_hess_analytic` recipe.
    fn analytic_shape_score_hessian(
        &self,
        _y: ndarray::ArrayView1<f64>,
        _mu: ndarray::ArrayView1<f64>,
        _dp: f64,
        _n_minus_mp: f64,
        _phi_score: f64,
    ) -> Option<ndarray::Array2<f64>> {
        None
    }

    /// Whether `PirlsInner` should use the Newton (observed-info) IRLS
    /// weights with per-row Fisher fallback rather than pure Fisher
    /// scoring. Default `false` — pure Fisher is correct for canonical-link
    /// exponential families (Gaussian + identity, Poisson + log, Bernoulli
    /// + logit, Gamma + reciprocal).
    ///
    /// Override on losses whose typical link is **non-canonical** (e.g.
    /// `InverseGaussian` + log, whose canonical link is `1/μ²`). For those
    /// pairs mgcv runs full Newton on the deviance with α = 1 + (y-μ)·(
    /// V'(μ)/V(μ) + g''(μ)/g'(μ)) and falls back to Fisher per-row when
    /// `α ≤ 0` (e.g. ~43% of obs for InverseGaussian + log). PIRLS at
    /// convergence has the same β̂ either way (both are stationary points
    /// of the penalised deviance), but `log|H| = log|X'WX + λS|` uses the
    /// post-convergence W — Fisher vs Newton differs there, and the gap
    /// flows through the REML score and into ρ̂. Reference: v0.x
    /// `src/pirls/mod.rs::Family::is_canonical_link` (lines 446-467) and
    /// `src/pirls/row_step.rs::compute_irls_wz`.
    fn use_newton_irls(&self) -> bool {
        false
    }

    /// Per-element initial μ for PIRLS. Default is mgcv's Bernoulli-style
    /// shrinkage `μ_i = (y_i + ȳ) / 2`. Family overrides:
    /// - **Poisson** (log link) → `max(y_i, 0.1)` to keep μ > 0 before
    ///   `link(μ) = log μ` is taken.
    /// - **Gamma** (log link) → `max(y_i, ε)` similarly.
    /// - **Bernoulli** (logit link) → `(y_i + 0.5) / 2` to keep μ ∈ (0, 1).
    ///
    /// The link is applied OUTSIDE this function (in PIRLS) — `initial_mu`
    /// returns μ on the natural scale; PIRLS then maps to η via `link.link()`.
    /// This keeps `Loss` link-agnostic.
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        let n = y.len();
        let y_bar: f64 = y.iter().sum::<f64>() / (n.max(1) as f64);
        y.iter().map(|&yi| (yi + y_bar) * 0.5).collect()
    }
}

/// Layer 2b — link function `g` with `η = g(μ)`.
pub trait Link {
    fn link(&self, mu: f64) -> f64;
    fn inverse_link(&self, eta: f64) -> f64;
    /// `dμ/dη`. Identity → 1. Logit: `μ(1-μ)`.
    fn d_inverse_link(&self, eta: f64) -> f64;
    /// `dη/dμ = g'(μ)`. Identity → 1. Logit: `1/(μ(1-μ))`. Used in the
    /// PIRLS working response `z = η + (y - μ) · g'(μ)`.
    fn d_link_dmu(&self, mu: f64) -> f64;
    /// `d²η/dμ² = g''(μ)`. Default `0.0` matches Identity. Log: `-1/μ²`.
    /// Logit: `(2μ - 1) / (μ(1-μ))²`. Used by the Newton IRLS path in
    /// `PirlsInner` to compute the curvature correction `α = 1 + (y-μ)·(
    /// V'(μ)/V(μ) + g''(μ)/g'(μ))` for non-canonical links (v0.x
    /// `src/pirls/row_step.rs::compute_irls_wz`, lines 67-82).
    fn d2_link_dmu(&self, _mu: f64) -> f64 {
        0.0
    }
    /// `d³η/dμ³ = g'''(μ)`. Default `0.0`. Log: `2/μ³`. Used by the Tk·KK'
    /// gradient correction in `EnvelopeScore` for non-canonical-link
    /// families, where the score's `log|H|` carries an explicit
    /// β-derivative through the W matrix (mirroring v0.x
    /// `src/reml/mod.rs::reml_gradient_mgcv_exact_ift_inner_at_beta`, the
    /// `a1[i]` computation around line 2089).
    fn d3_link_dmu(&self, _mu: f64) -> f64 {
        0.0
    }
    /// `True` when the link is the canonical link for the loss it's paired
    /// with — lets InnerSolver collapse Newton to Fisher. Default `false`
    /// (caller must opt in for each (Loss, Link) pair).
    fn is_canonical(&self) -> bool {
        false
    }
}

/// Layer 2c — `V(μ)`. Gaussian constant → 1. Binomial → `μ(1-μ)`.
pub trait VarianceFn {
    fn variance(&self, mu: f64) -> f64;
    /// `dV/dμ`. Defaults to `0.0` (Gaussian constant variance). Used by the
    /// Newton IRLS path in `PirlsInner` to compute the curvature correction
    /// `α = 1 + (y-μ)·(V'(μ)/V(μ) + g''(μ)/g'(μ))` for non-canonical-link
    /// families (v0.x `src/pirls/row_step.rs::compute_irls_wz`).
    fn d_variance(&self, _mu: f64) -> f64 {
        0.0
    }
    /// `d²V/dμ²`. Defaults to `0.0`. Used by the Tk·KK' gradient
    /// correction for non-canonical-link families — the `xx` term in
    /// v0.x `src/reml/mod.rs:2101` is `V''/V - (V'/V)² + g'''/g' -
    /// (g''/g')²`. For Inverse Gaussian (V=μ³): `V'' = 6μ`.
    fn d2_variance(&self, _mu: f64) -> f64 {
        0.0
    }
    /// Mirror of `Loss::set_shape_params`. For TDist's `TVariance` the
    /// variance σ² is one of the shape params and needs syncing with the
    /// Loss. Default no-op.
    fn set_shape_params(&mut self, _params: &[f64]) {}
}

/// Layer 3 — given fixed smoothing parameters θ, produce `β̂(θ)` + cached
/// linear-system pieces (Cholesky factor, fitted μ, RSS) for reuse by the
/// `ScoreDerivatives` layer. Concrete impls: `GaussianClosedFormInner`
/// (Phase 0, one Cholesky solve since identity link has no IRLS iteration);
/// later phases add `PirlsInner`, `FastRemlInner`, `ElfPirlsInner`,
/// `JointBetaPhiInner`.
///
/// **Score impls MUST go through this trait, not call concrete inner
/// solves directly.** This is the structural defence against "every new
/// family duplicates score.rs".
pub trait InnerSolver {
    /// What the solver returns. Phase 0 only needs the Gaussian
    /// `InnerFit`, but IRLS-iterative impls will widen this with the
    /// converged μ / η / working weights.
    type Fit;

    /// Solve the penalised inner system at the given log-λ vector.
    /// `rho.len() == 1` in Phase 0 (single smoothing parameter).
    fn fit(&self, rho: &Array1<f64>) -> Result<Self::Fit>;
}

/// Coordinate system the score reports in. Used by downstream consumers
/// (outer optimiser, vcov rebuild) to verify they're not comparing
/// quantities across mismatched bases — the structural defence against the
/// closed-form-vs-FD drift bug class described in plan §1.1 and
/// architecture-assumptions.md §D1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CoordsKind {
    /// Raw basis (no rotation applied).
    Identity,
    /// Basis is in rotated coords; the `&str` names the rotation (e.g.
    /// "stable" for `Sl.initial.repara`). Phase 0 doesn't use this.
    Reparametrised(&'static str),
}

/// Layer 4 — coupled `(value, grad, hess)` of the outer criterion at θ.
/// All three come from the same internal state — they CANNOT drift apart.
/// This is the keystone abstraction in the v2 architecture (plan §3.4):
/// the closed-form-vs-FD drift bug class is structurally impossible because
/// no caller can ask for grad and hess independently and get inconsistent
/// answers.
///
/// `coords()` reports the basis state — callers verify it before using
/// score outputs that depend on the basis (e.g., reconstructing vcov).
pub trait ScoreDerivatives {
    fn dim(&self) -> usize;

    /// What basis the score operates in. See `CoordsKind`.
    fn coords(&self) -> CoordsKind;

    fn value(&self, theta: &Array1<f64>) -> Result<f64>;

    /// Coupled `(value, grad)`. Cheaper than `value_grad_hess` when the
    /// caller doesn't need the Hessian (e.g., a gradient-only convergence
    /// check at iteration 0).
    fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)>;

    /// Coupled `(value, grad, hess)`. This is the surface the outer
    /// optimiser uses; callers MUST NOT FD-probe `value_and_grad` to get a
    /// Hessian (that re-creates the very bug class the trait was designed
    /// to prevent). If a `ScoreDerivatives` impl can't produce an analytic
    /// Hessian, it should FD internally — the FD happens with the impl's
    /// own knowledge of which σ²/coords/etc. to hold fixed.
    fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)>;
}

/// Result of an outer-loop optimisation.
pub struct OuterFit {
    pub theta: Array1<f64>,
    pub value: f64,
    pub grad_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Layer 5 — outer optimiser. Consumes only `ScoreDerivatives`; knows
/// nothing about Loss/Link/Basis/Family. Concrete impls plug-and-play:
/// `NewtonWithHalving` (Phase 0), `BfgsQuasiNewton`, `Brent1D`,
/// `JointBfgsTrustRegion`.
pub trait OuterSolver {
    fn minimize<S: ScoreDerivatives>(&self, score: &S, theta0: Array1<f64>) -> Result<OuterFit>;
}
