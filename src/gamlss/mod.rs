//! GAMLSS — generalised additive models for location, scale and shape:
//! multi-linear-predictor families where several distributional parameters
//! (location, scale, skewness, kurtosis, …) are each smooth functions of the
//! covariates and fit jointly.
//!
//! Built component-by-component, TDD-style:
//!   - phase 1 ([`shash`]): the sinh-arcsinh density and its per-observation
//!     derivatives in param space, validated against mgcv + finite differences.
//!   - phase 2 ([`shash`]): the link chain (`logeb` on τ) and η-space
//!     derivatives, validated the same way.
//!   - phase 3 ([`shash_init`]): the initialiser (ridge regressions for μ/τ,
//!     ε=φ=0), confronted with mgcv's `pen.reg` in its zero-penalty limit.
//! The joint multi-predictor inner solver, the coupled REML over per-block
//! smoothing parameters, and the fitted/predict surface follow in later phases.
//!
//! (The orthogonal Gaussian location-scale family `gaulss` lives in
//! `crate::fit::gaulss` — it needs no dense block machinery because its
//! Fisher information is block-diagonal; `shash` is the non-orthogonal case
//! this module is being built for.)

pub mod shash;
pub mod shash_init;
