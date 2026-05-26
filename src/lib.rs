//! gammon — GAM core, v2 architecture experiment.
//!
//! Phase 0 scope (per `Projects/mgcv_rust/plans/mgcv_rust - v2 Architecture
//! Plan 2026-05-22.md`): Gaussian + identity link + CR spline + sum-to-zero
//! constraint, single smooth, fit end-to-end through the trait stack with
//! minimal parity against mgcv on a few fixtures. No other families, no
//! tensor products, no Python bindings.
//!
//! Layering (top-down):
//!
//! ```text
//! Layer 1   Basis         (basis.rs)      — design matrix + penalties
//! Layer 1.5 BasisTransform (transform.rs) — sum-to-zero, etc.
//! Layer 2   Loss / Link / Variance (family.rs)
//! Layer 3   InnerSolver   (inner.rs)     — β̂(θ) given fixed θ
//! Layer 4   ScoreDerivatives (score.rs)  — REML/LAML criterion + grad
//! Layer 5   OuterSolver   (outer.rs)     — Newton on θ
//! Layer 6   FittedModel / Predictor (fit.rs)
//! ```

pub mod basis;
pub mod design;
pub mod error;
pub mod family;
pub mod fit;
pub mod inner;
pub mod outer;
pub mod score;
pub mod special;
pub mod traits;
pub mod transform;

#[cfg(feature = "python")]
pub mod python;

pub use design::{
    combined_s, Additive, AdditivePredictor, Cr, CrPredictor, CrStable, CrStablePredictor,
    DesignStrategy, Predictor, PreparedDesign, Re, RePredictor, TermSpec,
};
pub use error::{GammonError, Result};
pub use family::{ocat_init_theta, OcatLoss, OcatVariance};
pub use fit::{
    fit, fit_with, fit_with_design, fit_with_solver, FamilyFit, FamilyFitWithSolver,
    FitWithProfile, FittedGam, LinkKind, PredictScale,
};
pub use inner::{CholeskySolver, LinearSolver, LuSolver};
pub use score::{FixedAtOneProfile, MgcvTwoSigmaProfile, OwnedByLossProfile, Profile};
pub use transform::{StableReparam, SumToZero};
