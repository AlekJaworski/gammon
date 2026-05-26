//! Quantile / ELF (qgam-style) with identity link (Phase 10 — v0.2 port).

use crate::traits::{Loss, VarianceFn};

use super::link::IdentityLink;
use super::Family;

/// Smooth quantile likelihood — ELF (Extended Log-F) loss from qgam
/// (Fasiolo et al. 2021). Calibrated smoothing of the pinball loss
/// `ρ_τ(r) = max(τ·r, (τ-1)·r)`; the smoothed negative log-likelihood is
///
/// ```text
///   L(r; τ, σ, λ) = [λ(1-τ)log(1-τ) + λτ·log(τ) - (1-τ)r
///                    + λ·log(1 + exp(r/λ))] / σ
/// ```
///
/// where `r = y - μ` (identity link), `τ ∈ (0, 1)` is the user-supplied
/// target quantile, `σ > 0` is the likelihood scale, and `λ > 0` is
/// qgam's logistic width (`co`). As σ → 0, `2L → ρ_τ(r)`.
///
/// **Stateful loss with three shape params**: `τ` is user-supplied at
/// construction and NOT optimised; `(σ, λ)` are profiled. gammon's Phase-10
/// port follows v0.x's `fit_pirls_quantile` heuristic — both σ and λ are
/// derived from a Gaussian warm-start and fixed across the outer ρ-loop
/// (so `n_shape_params() = 0`). Fully profiling (σ, λ) via the outer
/// Newton is a future extension.
///
/// Variance is "constant" in μ (location-scale family) — `ElfVariance`
/// returns a μ-free sentinel scaled by σ²; the actual per-obs working
/// weights come from `ArmijoElfInner`, NOT from `V(μ)·g'(μ)²`.
#[derive(Clone)]
pub struct ElfLoss {
    /// Target quantile, τ ∈ (0, 1). User-supplied.
    pub tau: f64,
    /// Likelihood scale (qgam's `exp(theta)`). Set heuristically.
    pub sigma: f64,
    /// Logistic width (qgam's `co`). Set heuristically (often equal to σ).
    pub lambda: f64,
}

/// μ-free variance for ELF. Returns `4σ²` as a sentinel — `s(1-s) ≤ 1/4`
/// makes `V_max ≥ 4σ²` if anything ever reads it. The real per-obs
/// weights come from `ArmijoElfInner`.
#[derive(Clone)]
pub struct ElfVariance {
    pub sigma: f64,
}

/// Per-observation pieces of the ELF loss — same shape as v0.x's
/// `QuantileElfParts` but only the bits gammon needs.
///
/// `r = y - μ`, `u = r / λ`, `s = sigmoid(u)` (logistic CDF),
/// `softplus(u) = log(1 + exp(u))`. Deviance = `2L` per v0.x convention.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ElfParts {
    pub deviance: f64,
    /// `∂(L)/∂μ` — note this is the NLL derivative, not deviance.
    pub dl_dmu: f64,
    /// `∂²(L)/∂μ²` — always positive.
    pub d2l_dmu: f64,
    /// `s = sigmoid((y - μ) / λ)` — kept for the working weights/response.
    pub sigmoid: f64,
}

fn elf_sigmoid_stable(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn elf_softplus_stable(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Compute ELF per-obs pieces. Ported verbatim from v0.x
/// `src/pirls/mod.rs::quantile_elf_parts` (qgam `elf.R:122-167`).
pub(crate) fn elf_parts(y: f64, mu: f64, tau: f64, sigma: f64, lambda: f64) -> ElfParts {
    let r = y - mu;
    let u = r / lambda;
    let s = elf_sigmoid_stable(u);
    let softplus = elf_softplus_stable(u);
    // dl = s(1-s)/λ — derivative of logistic CDF.
    let dl = s * (1.0 - s) / lambda;
    // t = λ(1-τ)log(1-τ) + λτ·log(τ) - (1-τ)r + λ·softplus
    let t = (1.0 - tau) * lambda * (1.0 - tau).ln() + lambda * tau * tau.ln() - (1.0 - tau) * r
        + lambda * softplus;
    // v0.x: dmu_qgam = -2(s - 1 + τ)/σ — this is ∂(2L)/∂μ.
    let dmu_qgam = -2.0 * (s - 1.0 + tau) / sigma;
    let dmu2_qgam = 2.0 * dl / sigma;
    ElfParts {
        deviance: 2.0 * t / sigma,
        // Loss::d_loss_dmu returns ∂(deviance)/∂μ — match v0.x's dmu_qgam
        // (NOT v0.x's QuantileElfParts.dmu, which is 0.5×dmu_qgam — that
        // field stores ∂(L)/∂μ rather than ∂(deviance)/∂μ; gammon's
        // convention is deviance-based throughout, like Gaussian/Bernoulli).
        dl_dmu: dmu_qgam,
        d2l_dmu: dmu2_qgam,
        sigmoid: s,
    }
}

impl Loss for ElfLoss {
    /// μ₀ = y itself for the ELF/identity-link path. PIRLS-style
    /// `(y + ȳ)/2` shrinkage is wrong for quantile regression — at the
    /// τ-quantile target we don't want the mean-centred shrinkage. The
    /// actual warm-start used by `fit_quantile_cr` is a Gaussian-fit
    /// residual quantile shift; this default is only the fallback if
    /// some caller uses ElfLoss outside the gammon fit driver.
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        y.iter().copied().collect()
    }

    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        elf_parts(y, mu, self.tau, self.sigma, self.lambda).deviance
    }

    /// Saturated log-lik per qgam `elf.R:204-216`. The σ enters via
    /// `a = λ(1-τ)/σ`, `b = λτ/σ`. φ is fixed at 1 (σ is the family
    /// param, not external dispersion) so `_scale` is ignored.
    fn saturated_log_lik(&self, _y: f64, _scale: f64) -> f64 {
        let sigma = self.sigma.max(1e-12);
        let lambda = self.lambda.max(1e-12);
        let tau = self.tau;
        let a = lambda * (1.0 - tau) / sigma;
        let b = lambda * tau / sigma;
        // log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)
        let log_beta = crate::special::log_gamma(a) + crate::special::log_gamma(b)
            - crate::special::log_gamma(a + b);
        (1.0 - tau) * lambda * (1.0 - tau).ln() / sigma + lambda * tau * tau.ln() / sigma
            - lambda.ln()
            - log_beta
    }

    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        elf_parts(y, mu, self.tau, self.sigma, self.lambda).dl_dmu
    }

    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        elf_parts(y, mu, self.tau, self.sigma, self.lambda).d2l_dmu
    }

    fn fixed_dispersion(&self) -> Option<f64> {
        Some(1.0)
    }

    // No shape params exposed to the outer Newton. v0.x's `fit_pirls_quantile`
    // sets σ and λ heuristically from a Gaussian warm-start and never
    // updates them — gammon's Phase-10 port mirrors that. Profiling (σ, λ)
    // via outer Newton is a future extension.
    fn n_shape_params(&self) -> usize {
        0
    }
}

impl VarianceFn for ElfVariance {
    /// μ-free sentinel — `4σ²` matches v0.x's
    /// `Family::Quantile { sigma, .. } => 4.0 * sigma * sigma`. Not actually
    /// used by `ArmijoElfInner` (which computes per-obs weights from
    /// `dmu2_qgam` directly).
    fn variance(&self, _mu: f64) -> f64 {
        4.0 * self.sigma * self.sigma
    }
}

/// Phase 10 convenience constructor — ELF + identity link at given
/// (τ, σ, λ). σ and λ are typically set by `fit_quantile_cr` from a
/// data-driven heuristic; pass `1.0` for both as a placeholder.
pub fn elf_identity(
    tau: f64,
    sigma: f64,
    lambda: f64,
) -> Family<ElfLoss, IdentityLink, ElfVariance> {
    Family::new(
        ElfLoss { tau, sigma, lambda },
        IdentityLink,
        ElfVariance { sigma },
    )
}
