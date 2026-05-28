//! Phase 2 — Shape-aware envelope score for stateful Loss families.
//!
//! Single generic impl `ShapeAwareEnvelopeScore<L, K, V, B, P, S>` covers
//! both the PIRLS-driven families (TDist/scat, Tweedie, NegBin, Gamma in
//! shape-aware mode) AND the ocat family. The difference is purely in
//! WHICH inner solver to instantiate per probe — `ShapeInnerBuilder`
//! lifts that from a per-family duplicated body into a one-line
//! type-level choice.
//!
//! `S: LinearSolver` (default `CholeskySolver`) propagates the backend
//! choice from the inner-builder through to the score body's
//! `inner_fit.log_det_a()` / `inner_fit.trace_a_inv(...)` calls.
//!
//! Split layout (task #98):
//! - `builder` — `ShapeInnerBuilder` trait + `PirlsInnerBuilder` /
//!   `OcatInnerBuilder` unit-struct impls.
//! - `score`   — `ShapeAwareEnvelopeScore` struct, type aliases,
//!   `ScoreDerivatives` impl, and the orchestrating `fit_inner_at` /
//!   `score_value` helpers + `FrozenBetaCtx`.
//! - `gradient` — `compute_rho_envelope_gradient` (single source of
//!   truth), `compute_value`, `compute_value_grad`, `eval_grad_*`,
//!   `analytic_shape_grad_via_ift`.
//! - `hessian`  — `compute_value_grad_hess_analytical`, the FD-based
//!   Hessian variants.

mod builder;
mod gradient;
mod hessian;
mod score;

pub use builder::{OcatInnerBuilder, PirlsInnerBuilder, ShapeInnerBuilder};
pub use score::{
    ShapeAwareEnvelopeScore, ShapeAwareOcatScore, ShapeAwarePirlsScore,
    ShapeAwarePirlsScoreOwnedPhi,
};
