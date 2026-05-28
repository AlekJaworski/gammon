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
/// as just `σ²` and folds `ν/(ν-2)` into the working weights. gamrs mirrors
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

    /// Saturated log-lik per observation: `log Γ((ν+1)/2) - log Γ(ν/2)
    /// - 0.5 log(π·ν·σ²)` — independent of y (scat is location-scale, so
    /// the saturated density is constant in the response). The `_scale`
    /// arg is the external dispersion — TDist owns its scale via the
    /// shape param `self.sigma2`, so the external one is ignored.
    ///
    /// **Why both Γ terms matter under joint Newton on (λ, ν, σ²)**:
    /// historically Phase-2a dropped the Γ ratio with the rationale "Γ
    /// terms are constants in (ν, σ²)" — that is **false**: Γ((ν+1)/2)
    /// and Γ(ν/2) both move with ν, and the Σ_i ls_i term carries an
    /// n·(dlog Γ((ν+1)/2)/dν − dlog Γ(ν/2)/dν) component into the
    /// LAML gradient w.r.t. log(ν - 2). Dropping it (as Phase 2a did)
    /// made the outer Newton's ∂LAML/∂(log(ν - 2)) chase the wrong
    /// optimum, pulling ν toward the lower bound (ν → 2⁺ saturated at
    /// `log(ν - 2) = -10` on the multi-smooth synthetic) instead of
    /// the interior optimum mgcv finds at ν ≈ 5. Includes both Γ terms
    /// to match v0.x `pirls/mod.rs:521-528` (Family::TDist branch of
    /// `saturated_log_likelihood`) byte-for-byte at fixed (ν, σ²).
    fn saturated_log_lik(&self, _y: f64, _scale: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let half_nu_p1 = (self.nu + 1.0) / 2.0;
        let half_nu = self.nu / 2.0;
        crate::special::log_gamma(half_nu_p1) - crate::special::log_gamma(half_nu)
            - 0.5 * (pi * self.nu * self.sigma2).ln()
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
    /// mgcv `build_outer_search_vector`: TDistLogSigma2 step cap 1.0,
    /// TDistLogNu (log(ν-2)) step cap 1.0.
    fn shape_axis_step_caps(&self) -> Vec<f64> {
        vec![1.0, 1.0]
    }
    fn set_shape_params(&mut self, params: &[f64]) {
        debug_assert_eq!(params.len(), 2, "TDist expects 2 shape params");
        self.sigma2 = params[0].exp();
        self.nu = 2.0 + params[1].exp();
    }
    fn get_shape_params(&self) -> Vec<f64> {
        vec![self.sigma2.ln(), (self.nu - 2.0).ln()]
    }

    /// Match v0.x `fit_pirls_tdist`'s β-tolerance. v0.x's scat diagnostic
    /// call site (`lib.rs:1162`) and the regular outer-fit path both pass
    /// `1e-8` to `fit_pirls_tdist`. gamrs's PirlsOpts default of `1e-9`
    /// stops one decimal later than v0.x, leaving a residual β-gap that
    /// flows through Layer-3 into the score-formula's `log|H|`. Same
    /// convention as ocat (commit `4c95a72`).
    fn pirls_dev_rel_tol(&self) -> f64 {
        1.0e-8
    }

    /// EXPERIMENTAL — diagnostic toggle for the mgcv-style rank heuristic
    /// (centered CR(k) treated as rank k−2 by mgcv vs gamrs's k−1). Empirical
    /// check for scat parallel to ocat (commit `d91b710`).
    fn score_rank_adjustment(&self) -> i32 {
        -1
    }

    /// Provide Level-1 derivatives (`Dmu3, Dth, Dmuth, Dmu2th`) to the
    /// shape-aware score's analytic θ-gradient assembly. Mirrors ocat's
    /// `OcatLoss::level1_shape_derivatives` (commits `85946a1` + `c38083c`)
    /// — the IFT path in `score/shape_aware.rs::analytic_shape_grad_via_ift`
    /// and the Tk·KK' β-chain in `compute_rho_envelope_gradient` are
    /// family-agnostic; they fire as soon as the loss returns `Some(...)`.
    ///
    /// For scat the two shape params are `θ_0 = log σ²` and
    /// `θ_1 = log(ν − 2)`. All four arrays are analytic, derived from
    /// `D(y, μ; ν, σ²) = (ν+1) · log(1 + (y−μ)²/(ν·σ²))`.
    ///
    /// Notation in the derivation: `r = y − μ`, `q = ν·σ²`, `s = q + r²`.
    /// Identity link so `μ = η`. The shape-transform Jacobians are
    /// `∂σ²/∂θ_0 = σ²`, `∂q/∂θ_0 = q`, `∂(ν−2)/∂θ_1 = ν − 2`,
    /// `∂ν/∂θ_1 = ν − 2`, `∂q/∂θ_1 = σ²·(ν − 2)` (= `qs_theta1` below).
    ///
    /// The `dmu3` / `dth` / `dmuth` / `dmu2th` arrays already incorporate
    /// the per-row prior weight (same convention as ocat —
    /// `family/ocat.rs::ocat_dd_level1`; mgcv `efam.r:2814-2832`).
    fn level1_shape_derivatives(
        &self,
        y: ndarray::ArrayView1<f64>,
        eta: ndarray::ArrayView1<f64>,
        prior_w: Option<ndarray::ArrayView1<f64>>,
    ) -> Option<crate::traits::Level1ShapeDerivs> {
        use ndarray::{Array1, Array2};
        let n = y.len();
        let nu = self.nu;
        let sigma2 = self.sigma2;
        let nu_p1 = nu + 1.0;
        let nu_minus_2 = nu - 2.0;
        let qs_theta1 = sigma2 * nu_minus_2; // ∂q/∂θ_1
        let q = nu * sigma2; // ν·σ² — constant across rows

        let mut dmu3 = Array1::<f64>::zeros(n);
        let mut dth = Array2::<f64>::zeros((n, 2));
        let mut dmuth = Array2::<f64>::zeros((n, 2));
        let mut dmu2th = Array2::<f64>::zeros((n, 2));

        for i in 0..n {
            let r = y[i] - eta[i];
            let r2 = r * r;
            let s = q + r2;
            let s2 = s * s;
            let s3 = s2 * s;
            let wt_i = prior_w.map(|w| w[i]).unwrap_or(1.0);

            // ∂³D/∂μ³ = 4·r·(ν+1)·(3q − r²) / s³. Includes wt (ocat
            // convention — IFT consumer pre-applies wt at this step).
            dmu3[i] = wt_i * 4.0 * r * nu_p1 * (3.0 * q - r2) / s3;

            // ── θ_0 = log σ² ────────────────────────────────────────────
            // ∂D/∂θ_0 = (ν+1)·(q/s − 1) = −(ν+1)·r² / s.
            dth[[i, 0]] = wt_i * (-nu_p1 * r2 / s);
            // ∂(∂D/∂μ)/∂θ_0 = 2(ν+1)·r·q / s².
            dmuth[[i, 0]] = wt_i * (2.0 * nu_p1 * r * q / s2);
            // ∂(∂²D/∂μ²)/∂θ_0 = 2(ν+1)·q·(3r² − q) / s³.
            dmu2th[[i, 0]] = wt_i * (2.0 * nu_p1 * q * (3.0 * r2 - q) / s3);

            // ── θ_1 = log(ν − 2) ────────────────────────────────────────
            // ∂D/∂θ_1 = (ν−2)·[log(1 + r²/q) − (ν+1)·r²/(ν·s)].
            let log_term = if q > 0.0 { (1.0 + r2 / q).ln() } else { 0.0 };
            dth[[i, 1]] = wt_i * nu_minus_2 * (log_term - nu_p1 * r2 / (nu * s));
            // ∂(∂D/∂μ)/∂θ_1 = −2r·[(ν−2)·s − (ν+1)·qs_theta1] / s².
            dmuth[[i, 1]] =
                wt_i * (-2.0 * r * (nu_minus_2 * s - nu_p1 * qs_theta1) / s2);
            // ∂(∂²D/∂μ²)/∂θ_1
            //   = 2·[(ν−2)·(q − r²)·s + (ν+1)·qs_theta1·(3r² − q)] / s³.
            dmu2th[[i, 1]] = wt_i
                * (2.0
                    * (nu_minus_2 * (q - r2) * s + nu_p1 * qs_theta1 * (3.0 * r2 - q))
                    / s3);
        }

        Some(crate::traits::Level1ShapeDerivs {
            dmu3,
            dth,
            dmuth,
            dmu2th,
        })
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
