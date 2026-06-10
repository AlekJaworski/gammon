//! GAMLSS — generalised additive models for location, scale and shape:
//! multi-linear-predictor families where several distributional parameters
//! (location, scale, skewness, kurtosis, …) are each smooth functions of the
//! covariates and fit jointly.
//!
//! Built component-by-component, TDD-style. Phase 1 (here): the `shash`
//! (sinh-arcsinh) density and its per-observation derivatives, validated
//! against mgcv and finite differences. The joint multi-predictor inner
//! solver, the coupled REML over per-block smoothing parameters, and the
//! fitted/predict surface follow in later phases.
//!
//! (The orthogonal Gaussian location-scale family `gaulss` lives in
//! `crate::fit::gaulss` — it needs no dense block machinery because its
//! Fisher information is block-diagonal; `shash` is the non-orthogonal case
//! this module is being built for.)

pub mod shash;
