//! Scaled-t (TDist / mgcv `scat`) with identity link.

use crate::traits::{Loss, VarianceFn};

use super::link::IdentityLink;
use super::Family;

/// Heavy-tailed scaled-t likelihood: `y_i ~ μ_i + σ · T_ν` where `T_ν` is
/// a standard Student-t with `ν` degrees of freedom. Used for robust
/// regression where Gaussian noise mis-specifies the tails.
///
/// **Stateful loss** — `nu` and `sigma2` are SHAPE PARAMETERS of the
/// family, not data. In mgcv they're jointly optimised with the smoothing
/// parameter λ via an outer Newton over `[log λ, log σ², ν-transform]`.
///
/// Phase 2a ships TDist with `nu` / `sigma2` as struct fields fixed at
/// construction time. Phase 2b will extend the outer optimiser to handle
/// multi-θ so the shape params can be joint-optimised (see
/// architecture-assumptions.md §E for the plan).
#[derive(Clone)]
pub struct TDist {
    /// Degrees of freedom. mgcv requires ν > 2 for finite variance; we
    /// don't enforce here (PIRLS handles ν ∈ (1, 2] in principle, just
    /// with slow tails).
    pub nu: f64,
    /// Squared scale parameter. The actual t-scale is √σ². Plays the same
    /// role as Gaussian σ² but is internal to the family — mgcv's
    /// dispersion `scale` stays at 1 for scat.
    pub sigma2: f64,
}

/// Variance function for TDist (location-scale family). The "variance" is
/// constant `ν·σ²/(ν-2)` for finite ν > 2, OR equivalently mgcv treats it
/// as just `σ²` and folds `ν/(ν-2)` into the working weights. gammon mirrors
/// mgcv's convention: `V(μ) = σ²` and the PIRLS working weights use the
/// t-specific `∂²L/∂μ²` directly via `Loss::d2_loss_dmu`.
#[derive(Clone)]
pub struct TVariance {
    pub sigma2: f64,
}

impl Loss for TDist {
    /// `D_i = (ν+1) · log(1 + (y-μ)² / (ν·σ²))` per mgcv `scat$dev.resids`.
    fn deviance_per_obs(&self, y: f64, mu: f64) -> f64 {
        let r = y - mu;
        (self.nu + 1.0) * (1.0 + r * r / (self.nu * self.sigma2)).ln()
    }

    /// Saturated log-lik: `log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(π·ν·σ²)`
    /// — independent of y. We drop the y-dependent piece since the score
    /// formula is invariant under additive constants in y; the *λ*-gradient
    /// is unaffected. Phase 2a only uses `saturated_log_lik` inside the
    /// score's `Σ ls(y)` sum, where this constant is fine. The `_scale`
    /// arg is the external dispersion — TDist owns its scale via the
    /// shape param `self.sigma2`, so the external one is ignored.
    fn saturated_log_lik(&self, _y: f64, _scale: f64) -> f64 {
        // -½ log(π·ν·σ²) + log Γ((ν+1)/2) - log Γ(ν/2)
        // The Γ terms are constants in (ν, σ²); we omit them (constant in
        // y, doesn't affect the λ-gradient — same convention Gaussian uses).
        // For full joint-θ optimisation Phase 2b will reinstate them.
        let pi = std::f64::consts::PI;
        -0.5 * (pi * self.nu * self.sigma2).ln()
    }

    /// `∂D/∂μ = -2(ν+1)·(y-μ) / (ν·σ² + (y-μ)²)`.
    fn d_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let r = y - mu;
        -2.0 * (self.nu + 1.0) * r / (self.nu * self.sigma2 + r * r)
    }

    /// `∂²D/∂μ² = 2(ν+1)·(ν·σ² - r²) / (ν·σ² + r²)²` where `r = y - μ`.
    /// Positive for `|r| < √(ν·σ²)` (the "core" of the distribution) and
    /// negative for outliers — this is what gives scat its robustness.
    fn d2_loss_dmu(&self, y: f64, mu: f64) -> f64 {
        let r = y - mu;
        let denom = self.nu * self.sigma2 + r * r;
        2.0 * (self.nu + 1.0) * (self.nu * self.sigma2 - r * r) / (denom * denom)
    }

    /// Dispersion φ = 1 for scat — the actual scale lives inside the
    /// family as `sigma2`. Same convention as mgcv.
    fn fixed_dispersion(&self) -> Option<f64> {
        Some(1.0)
    }

    /// scat owns 2 shape params: `[log σ², log(ν - ν_min)]` with
    /// `ν_min = 2.0` to keep ν > 2 (finite variance). Transform `log(ν-2)`
    /// is mgcv's choice (`gam.fit5.r`).
    fn n_shape_params(&self) -> usize {
        2
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        debug_assert_eq!(params.len(), 2, "TDist expects 2 shape params");
        self.sigma2 = params[0].exp();
        self.nu = 2.0 + params[1].exp();
    }
    fn get_shape_params(&self) -> Vec<f64> {
        vec![self.sigma2.ln(), (self.nu - 2.0).ln()]
    }
}

impl VarianceFn for TVariance {
    /// Constant variance σ² (NOT μ-dependent — t is location-scale).
    fn variance(&self, _mu: f64) -> f64 {
        self.sigma2
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        // Sync `sigma2` from the first transformed param (log σ²). Ignores
        // the ν transform — variance doesn't depend on ν for scat.
        debug_assert_eq!(
            params.len(),
            2,
            "TVariance expects 2 shape params (slot 0 is log σ²)"
        );
        self.sigma2 = params[0].exp();
    }
}

/// Phase 2a convenience constructor — TDist + identity link at given shape.
pub fn tdist_identity(nu: f64, sigma2: f64) -> Family<TDist, IdentityLink, TVariance> {
    Family::new(TDist { nu, sigma2 }, IdentityLink, TVariance { sigma2 })
}
