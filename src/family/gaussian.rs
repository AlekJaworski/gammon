//! Gaussian Loss + constant variance.
//!
//! `deviance_per_obs(y, μ) = (y - μ)²` and `saturated_log_lik(y)` is a
//! constant in y (the σ² in `-½ log(2π σ²)` is profiled out in the REML
//! score for Gaussian, so we drop the constants here — the criterion is
//! invariant to them).

use crate::traits::{Loss, VarianceFn};

use super::link::IdentityLink;
use super::Family;

#[derive(Clone)]
pub struct Gaussian;

#[derive(Clone)]
pub struct ConstantVariance;

impl Loss for Gaussian {
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let r = y - mu;
        r * r
    }
    /// Gaussian saturated log-lik per obs at μ=y: `-0.5·log(2πφ)` (the y²/φ
    /// piece is y-only at saturation and is dropped — same convention as
    /// v0.x's `Family::Gaussian` branch in `pirls::saturated_log_likelihood`,
    /// which sums to `-n/2·log(2πφ)`). Phase-2b v0.2 port: reinstated so the
    /// new Form B score body (`-Mp/2·log(2πφ) - Σls`) gives the same value
    /// as Form A (`+(n-Mp)/2·log(2πφ)`) — `-Mp/2·log + n/2·log = (n-Mp)/2·log`.
    fn saturated_log_lik(&self, _y: f64, scale: f64) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        -0.5 * (two_pi * scale).ln()
    }
    // `d_loss_dmu` / `d2_loss_dmu` use the trait defaults (Gaussian).
    // `fixed_dispersion` returns `None` (σ² is profiled).
}

impl VarianceFn for ConstantVariance {
    fn variance(&self, _mu: f64) -> f64 {
        1.0
    }
}

/// Phase 0 convenience constructor — the only Family in gamrs so far.
pub fn gaussian_identity() -> Family<Gaussian, IdentityLink, ConstantVariance> {
    Family::new(Gaussian, IdentityLink, ConstantVariance)
}
