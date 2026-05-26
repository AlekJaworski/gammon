//! Layer 4 — closed-form REML / LAML score, generic over `Loss` family.
//!
//! Single formula handles both:
//!
//! 1. **Profiled σ²** families (Gaussian, Gamma, InverseGaussian) — score
//!    profiles σ̂² out of the marginal likelihood.
//! 2. **Known σ² = 1** families (Bernoulli, Poisson, NegBin) — the
//!    `(n-Mp)/2·log(2πσ²)` term drops, and σ² disappears from the
//!    envelope gradient's `λβ'Sβ/(2σ²)`.
//!
//! Per mgcv `gam.fit3.r:616-617`:
//!
//! ```text
//!   REML(ρ) = Dp/(2σ²) + (n - Mp)/2·log(2πσ²) + log|H|/2
//!             - log|λS|+/2 - Σ ls(y_i)
//! ```
//!
//! where `Dp = D + λβ'Sβ`, `D = Σ w_i · D_i(y_i, μ_i)` is the GLM
//! deviance, `H = X' diag(W_working) X + λS`, and `ls(y_i)` is
//! `Loss::saturated_log_lik`. For known-σ² families the `(n-Mp)/2·log(2πσ²)`
//! term is a constant in λ — we still include it so the reported score is
//! commensurable with mgcv (but its log-gradient is zero so it doesn't
//! affect the optimum).
//!
//! Split layout: this module re-exports the public surface; the actual
//! impls live in:
//! - `profile`      — `Profile` trait + `MgcvTwoSigmaProfile` /
//!                    `FixedAtOneProfile`.
//! - `envelope`     — `EnvelopeScore` (1-D outer Newton, no shape params).
//! - `shape_aware`  — `ShapeAwareEnvelopeScore`, `ShapeInnerBuilder` trait
//!                    + `PirlsInnerBuilder` / `OcatInnerBuilder`.
//!
//! `log|H|` and `tr(H⁻¹·M)` now live as methods on `GaussianInnerFit<S>`
//! — the previous standalone `trace_solve` here and `trace_a_inv_s` in
//! `inner/` (audit finding #4 — two impls of the same op) collapsed into
//! `LinearSolver::trace_a_inv`, dispatched via the `S` backend type.

pub mod envelope;
pub mod profile;
pub mod shape_aware;

pub use envelope::{EnvelopeScore, GaussianClosedFormScore};
pub use profile::{FixedAtOneProfile, MgcvTwoSigmaProfile, OwnedByLossProfile, Profile};
pub use shape_aware::{
    OcatInnerBuilder, PirlsInnerBuilder, ShapeAwareEnvelopeScore, ShapeAwareOcatScore,
    ShapeAwarePirlsScore, ShapeAwarePirlsScoreOwnedPhi, ShapeInnerBuilder,
};
