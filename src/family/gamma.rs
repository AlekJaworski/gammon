//! Gamma with log link (incl. `profile_score_sigma2` Newton).

use crate::traits::{Loss, VarianceFn};

use super::link::LogLink;
use super::Family;

/// Gamma likelihood for continuous positive responses (waiting times,
/// concentrations, etc.). Canonical link is `1/μ`; mgcv's `Gamma()` and
/// `Gamma(link="log")` are both supported but gammon ships only the log
/// link for now (most common in practice).
///
/// - Deviance: `D(y, μ) = 2[(y - μ)/μ - log(y/μ)]` for `y > 0`.
/// - Variance: `V(μ) = μ²`.
/// - Dispersion: σ² profiled.
/// - Saturated log-lik: dropped (lgamma/digamma terms in φ are score-relevant
///   for mgcv-exact parity, but the simpler form still gives correct ρ̂; see
///   architecture-assumptions.md §C4-gamma for the gap).
#[derive(Clone)]
pub struct Gamma;

#[derive(Clone)]
pub struct GammaVariance;

impl Loss for Gamma {
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        // Gamma μ > 0. Use y itself with a small floor.
        y.iter().map(|&yi| yi.max(1e-3)).collect()
    }

    /// `D(y, μ) = 2[(y - μ)/μ + log(μ/y)]` for `y > 0`.
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-300);
        let y = y.max(1e-300);
        2.0 * ((y - mu) / mu + (mu / y).ln())
    }

    /// Saturated log-lik per obs at μ=y: `-lgamma(1/φ) - log(φ)/φ - 1/φ - log(y)`.
    /// Sum over y gives `n·k(φ) - Σlog(y)` with `k(φ) = -lgamma(1/φ) - log(φ)/φ - 1/φ`.
    ///
    /// Phase-2b v0.2 port (2026-05-24): reinstated the φ-dependent `k(φ)`
    /// piece (previously dropped because the envelope-gradient formula
    /// wasn't σ²-chain aware). Now self-consistent: `profile_score_sigma2`
    /// uses the matching mgcv Newton root, so the score body's σ² satisfies
    /// `∂REML/∂σ²|_{σ²=σ²̂} = 0`, and the envelope gradient at σ²̂ has no
    /// chain-correction term to add. Mirrors v0.x
    /// `src/pirls/mod.rs::Family::Gamma | GammaLog` (lines 513-518).
    fn saturated_log_lik(&self, y: f64, scale: f64) -> f64 {
        let inv_phi = 1.0 / scale;
        let k = -crate::special::log_gamma(inv_phi) - scale.ln() * inv_phi - inv_phi;
        k - y.max(1e-300).ln()
    }

    /// `∂D/∂μ = 2(μ - y) / μ²`.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-300);
        2.0 * (mu - y) / (mu * mu)
    }

    /// `∂²D/∂μ² = 2(2y - μ) / μ³`.
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let mu = mu.max(1e-300);
        2.0 * (2.0 * y - mu) / (mu * mu * mu)
    }

    fn fixed_dispersion(&self) -> Option<f64> {
        None // PROFILED
    }

    /// Gamma's profile-φ Newton — root-find on the σ²-derivative of the
    /// REML score (Form B / GamFit3):
    ///
    /// ```text
    ///   F(φ) = dp + 2n·[ψ(1/φ) + log φ] + Mp·φ = 0
    ///   F'(φ) = (2n/φ)·[1 - ψ'(1/φ)/φ] + Mp
    /// ```
    ///
    /// Verbatim port from v0.x `src/pirls/mod.rs::Family::estimate_phi_mgcv`
    /// Gamma branch (lines 847-883). Triple guard: per-step relative damping
    /// `max(*0.1).min(*10)`, absolute floor `1e-8`, NaN trap.
    fn profile_score_sigma2(
        &self,
        dp: f64,
        n_obs: usize,
        n_minus_mp: f64,
        phi_init: f64,
    ) -> f64 {
        let n = n_obs as f64;
        let mp_f = n - n_minus_mp;
        let mut phi = phi_init.max(1e-8);
        let tol_abs = 1e-10 * (dp.abs() + mp_f + 1.0);
        for _ in 0..30 {
            let inv_phi = 1.0 / phi;
            let f = dp
                + 2.0 * n * (crate::special::digamma(inv_phi) + phi.ln())
                + mp_f * phi;
            if f.abs() < tol_abs {
                break;
            }
            let fp = (2.0 * n / phi)
                * (1.0 - crate::special::trigamma(inv_phi) * inv_phi)
                + mp_f;
            // Guard against zero / near-zero derivative.
            if fp.abs() < 1e-15 {
                break;
            }
            let delta = -f / fp;
            // Triple guard: relative damping + absolute floor + NaN trap.
            let phi_new = (phi + delta).max(phi * 0.1).min(phi * 10.0).max(1e-8);
            if !phi_new.is_finite() {
                break;
            }
            let converged = (phi_new - phi).abs() < 1e-12 * phi;
            phi = phi_new;
            if converged {
                break;
            }
        }
        phi
    }
}

impl VarianceFn for GammaVariance {
    fn variance(&self, mu: f64) -> f64 {
        mu * mu
    }
}

/// Phase 7 convenience constructor — Gamma + log link.
pub fn gamma_log() -> Family<Gamma, LogLink, GammaVariance> {
    Family::new(Gamma, LogLink, GammaVariance)
}
