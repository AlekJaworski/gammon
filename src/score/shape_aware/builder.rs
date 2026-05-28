//! `ShapeInnerBuilder` trait + canonical `PirlsInnerBuilder` /
//! `OcatInnerBuilder` impls. Lets `ShapeAwareEnvelopeScore` pick its
//! inner solver at the type level rather than via per-family score
//! duplication.

use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::family::{Family, IdentityLink, OcatLoss, OcatVariance};
use crate::inner::{
    CholeskySolver, GaussianInnerFit, LinearSolver, OcatInner, PirlsInner, PirlsOpts,
};
use crate::traits::{InnerSolver, Link, Loss, VarianceFn};

/// Build an `InnerSolver` from a freshly shape-synced family + the score's
/// owned design fields. The score body uses this to rebuild the inner per
/// outer probe (shape params shift every probe; the inner must see the
/// current family).
///
/// Two concrete impls in gamrs:
/// - `PirlsInnerBuilder<S>` (generic over `L, K, V, S`) — drives TDist/scat,
///   Tweedie, NegBin via the standard PIRLS loop.
/// - `OcatInnerBuilder<S>` — drives the ocat extended family via `OcatInner`,
///   constrained to `Family<OcatLoss, IdentityLink, OcatVariance>` (the
///   only valid Loss/Link/Variance triple for ordered categorical).
pub trait ShapeInnerBuilder<
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver = CholeskySolver,
>
{
    type Inner: InnerSolver<Fit = GaussianInnerFit<S>>;

    fn build(
        &self,
        family: Family<L, K, V>,
        x_design: Array2<f64>,
        y: Array1<f64>,
        prior_weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
        opts: PirlsOpts,
    ) -> Self::Inner;
}

/// Unit-struct builder for `PirlsInner<L, K, V, S>`. The `S` parameter
/// is resolved from the score's `ShapeAwareEnvelopeScore<…, S>` type
/// context — the builder itself stays stateless.
#[derive(Clone, Copy, Default)]
pub struct PirlsInnerBuilder;

impl<L, K, V, S> ShapeInnerBuilder<L, K, V, S> for PirlsInnerBuilder
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    S: LinearSolver,
{
    type Inner = PirlsInner<L, K, V, S>;

    fn build(
        &self,
        family: Family<L, K, V>,
        x_design: Array2<f64>,
        y: Array1<f64>,
        prior_weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
        opts: PirlsOpts,
    ) -> Self::Inner {
        PirlsInner {
            x_design,
            y,
            prior_weights,
            s_list,
            family,
            opts,
            _solver: PhantomData,
        }
    }
}

/// Unit-struct builder for `OcatInner<S>`. The `S` parameter is resolved
/// from the score's `ShapeAwareEnvelopeScore<…, S>` type context.
#[derive(Clone, Copy, Default)]
pub struct OcatInnerBuilder;

impl<S: LinearSolver> ShapeInnerBuilder<OcatLoss, IdentityLink, OcatVariance, S>
    for OcatInnerBuilder
{
    type Inner = OcatInner<S>;

    fn build(
        &self,
        family: Family<OcatLoss, IdentityLink, OcatVariance>,
        x_design: Array2<f64>,
        y: Array1<f64>,
        prior_weights: Option<Array1<f64>>,
        s_list: Vec<Array2<f64>>,
        opts: PirlsOpts,
    ) -> Self::Inner {
        OcatInner {
            x_design,
            y,
            prior_weights,
            s_list,
            family,
            opts,
            _solver: PhantomData,
        }
    }
}
