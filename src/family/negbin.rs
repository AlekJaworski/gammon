//! Negative Binomial with log link.

use crate::traits::{Loss, VarianceFn};

use super::link::LogLink;
use super::Family;

/// Negative binomial likelihood for over-dispersed counts. mgcv's `nb()`:
/// `V(μ) = μ + μ²/θ` (Poisson-like + quadratic over-dispersion term).
/// Canonical link is `log(μ/(μ+θ))` but mgcv uses `log` by convention
/// (non-canonical but standard).
///
/// One shape parameter `θ > 0` (over-dispersion). Transform: `log θ`.
/// θ small → heavy over-dispersion (variance dominated by μ²/θ); θ → ∞
/// recovers Poisson.
#[derive(Clone)]
pub struct NegBin {
    pub theta: f64,
}

/// μ-dependent variance for NegBin: `V(μ) = μ + μ²/θ`. `θ` must be kept in
/// sync with the Loss via `set_shape_params`.
#[derive(Clone)]
pub struct NegBinVariance {
    pub theta: f64,
}

impl Loss for NegBin {
    fn initial_mu(&self, y: ndarray::ArrayView1<f64>) -> ndarray::Array1<f64> {
        // Same `max(y, 0.1)` as Poisson — keeps log-link domain valid.
        y.iter().map(|&yi| yi.max(0.1)).collect()
    }

    /// `D(y, μ) = 2[y·log(max(1,y)/μ) - (y+θ)·log((y+θ)/(μ+θ))]`. mgcv
    /// `negbin$dev.resids`. For y=0: `D = 2θ·log((μ+θ)/θ)`.
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let theta = self.theta;
        let mu = mu.max(1e-300);
        if y > 0.0 {
            2.0 * (y * (y / mu).ln() - (y + theta) * ((y + theta) / (mu + theta)).ln())
        } else {
            2.0 * theta * ((mu + theta) / theta).ln()
        }
    }

    /// Saturated log-lik at μ=y: `lgamma(y+θ) - lgamma(θ) - lgamma(y+1)
    /// + y·log(y/(y+θ)) + θ·log(θ/(y+θ))`. The lgamma(y+1) term is
    /// constant in θ — we drop it for the same reason as Poisson. The
    /// `lgamma(y+θ) - lgamma(θ)` terms are kept because they depend on θ
    /// and contribute to the joint outer Newton's θ-gradient. φ is fixed
    /// at 1 (NegBin's dispersion lives entirely in θ) so `_scale` is moot.
    fn saturated_log_lik(&self, y: f64, _scale: f64) -> f64 {
        let theta = self.theta;
        let yt = y + theta;
        let lg = crate::special::log_gamma(yt) - crate::special::log_gamma(theta);
        let y_term = if y > 0.0 { y * (y / yt).ln() } else { 0.0 };
        let t_term = theta * (theta / yt).ln();
        lg + y_term + t_term
    }

    /// `∂D/∂μ = 2θ(μ - y) / [μ(μ + θ)]`. Unified across y=0 and y>0.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let theta = self.theta;
        let mu = mu.max(1e-300);
        2.0 * theta * (mu - y) / (mu * (mu + theta))
    }

    /// `∂²D/∂μ² = 2θ · [-μ² + 2yμ + yθ] / [μ²(μ + θ)²]`.
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let theta = self.theta;
        let mu = mu.max(1e-300);
        let num = -mu * mu + 2.0 * y * mu + y * theta;
        let denom = mu * mu * (mu + theta) * (mu + theta);
        2.0 * theta * num / denom
    }

    fn fixed_dispersion(&self) -> Option<f64> {
        Some(1.0) // σ² fixed; θ is the shape param, not φ.
    }

    fn n_shape_params(&self) -> usize {
        1
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        debug_assert_eq!(params.len(), 1, "NegBin expects 1 shape param (log θ)");
        self.theta = params[0].exp();
    }
    fn get_shape_params(&self) -> Vec<f64> {
        vec![self.theta.ln()]
    }
}

impl VarianceFn for NegBinVariance {
    fn variance(&self, mu: f64) -> f64 {
        mu + mu * mu / self.theta
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        debug_assert_eq!(
            params.len(),
            1,
            "NegBinVariance expects 1 shape param (log θ)"
        );
        self.theta = params[0].exp();
    }
}

/// Phase 6 convenience constructor — NegBin + log link at given θ₀.
pub fn negbin_log(theta: f64) -> Family<NegBin, LogLink, NegBinVariance> {
    Family::new(NegBin { theta }, LogLink, NegBinVariance { theta })
}
