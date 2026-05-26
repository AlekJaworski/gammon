//! Generic fit drivers — collapse the per-family boilerplate.
//!
//! Each driver is generic over `S: LinearSolver` (default `CholeskySolver`
//! at the public-entry-point level). The S parameter flows from the
//! caller's choice of `gammon::fit::<_, _, _, LuSolver>(...)` through to
//! the inner solver and ultimately the emitted `GaussianInnerFit<S>`.

use std::marker::PhantomData;

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::design::PreparedDesign;
use crate::error::Result;
use crate::family::Family;
use crate::inner::{CholeskySolver, GaussianInnerFit, LinearSolver, PirlsInner, PirlsOpts};
use crate::outer::{NewtonOpts, NewtonWithHalving};
use crate::score::{EnvelopeScore, Profile, ShapeAwareEnvelopeScore, ShapeInnerBuilder};
use crate::traits::{CoordsKind, InnerSolver, Link, Loss, OuterSolver, VarianceFn};

use super::{compute_edf, compute_edf_per_term, compute_vcov, FittedGam, LinkKind};

/// Build a Pearson-φ̂ `scale_fn` closure for `fit_pirls_envelope`. The
/// returned closure captures `(y, prior_weights, n, mu_floor, V(μ))` and
/// computes `Σ wᵢ·(yᵢ-μᵢ)² / V(μᵢ)  /  (n - edf)`. Returns `NaN` if
/// `n - edf ≤ 0`. Eliminates the boilerplate copy-paste in the four
/// profiled-φ GLM families (QuasiPoisson, QuasiBinomial, Gamma, IG).
///
/// Generic over `S: LinearSolver` so the closure signature lines up
/// with the `S`-parameterised `GaussianInnerFit<S>` it consumes.
pub(crate) fn make_pearson_scale_fn<'a, S, F>(
    y_owned: Array1<f64>,
    prior_owned: Option<Array1<f64>>,
    n: usize,
    mu_floor: f64,
    variance_fn: F,
) -> impl FnOnce(&GaussianInnerFit<S>, f64) -> f64 + 'a
where
    S: LinearSolver + 'a,
    F: Fn(f64) -> f64 + 'a,
{
    move |fit, edf| {
        let n_minus_edf = (n as f64) - edf;
        if n_minus_edf <= 0.0 {
            return f64::NAN;
        }
        let sum: f64 = (0..n)
            .map(|i| {
                let mu_i = fit.mu[i].max(mu_floor);
                let r = y_owned[i] - mu_i;
                let w_i = prior_owned.as_ref().map(|w| w[i]).unwrap_or(1.0);
                w_i * r * r / variance_fn(mu_i)
            })
            .sum();
        sum / n_minus_edf
    }
}

/// Outer-Newton + final-fit + EDF for the "PIRLS + EnvelopeScore" stack.
/// Used by every family that fits this pattern (Bernoulli, Poisson,
/// QuasiPoisson, QuasiBinomial, Gamma, InverseGaussian). The 4 family-
/// specific bits stay in the public entry point: y-validation, the
/// `Family<L,K,V>` constructor, the `(LossMarker, Profile)` pair, and the
/// `scale_fn` (Pearson φ̂ or fixed 1.0).
///
/// Returns the assembled `FittedGam` directly so call sites are 3-4 lines.
///
/// `S: LinearSolver` defaults to `CholeskySolver` — pass an explicit
/// turbofish (`fit_pirls_envelope::<_, _, _, _, _, LuSolver, _>(...)`) to
/// switch to LU.
pub(crate) fn fit_pirls_envelope<L, K, V, M, P, S, SF>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    family: Family<L, K, V>,
    loss_marker: M,
    profile: P,
    scale_fn: SF,
    link_kind: LinkKind,
) -> Result<FittedGam>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    M: Loss,
    P: Profile<M>,
    S: LinearSolver,
    SF: FnOnce(&GaussianInnerFit<S>, f64) -> f64,
{
    let prior = prior_weights.map(|w| w.to_owned());
    let n = x.nrows();
    let n_terms = prep.s_list.len();
    let pirls: PirlsInner<L, K, V, S> = PirlsInner {
        x_design: prep.x_design.clone(),
        y: y.to_owned(),
        prior_weights: prior,
        s_list: prep.s_list.clone(),
        family,
        opts: PirlsOpts::default(),
        _solver: PhantomData,
    };
    let score = EnvelopeScore::<M, PirlsInner<L, K, V, S>, P, S>::with_inner(
        pirls,
        loss_marker,
        profile,
        y.to_owned(),
        prep.s_list.clone(),
        prep.rank_s_list.clone(),
        prep.mp,
        prep.log_pseudo_det_s_list.clone(),
    );
    let outer_solver = NewtonWithHalving::new(NewtonOpts::default());
    let outer = outer_solver.minimize(&score, Array1::zeros(n_terms))?;

    // Reuse the score's inner solver for the final fit (closes audit §B4).
    let final_fit: GaussianInnerFit<S> = score.inner.fit(&outer.theta)?;
    let edf = compute_edf(&prep.x_design, &final_fit.working_weights, &final_fit);
    let scale = scale_fn(&final_fit, edf);
    let vcov = compute_vcov(&final_fit, scale);
    let rho_hat = outer.theta.clone();
    let lambda_hat: Array1<f64> = rho_hat.iter().map(|&r| r.exp()).collect();
    let edf_per_term =
        compute_edf_per_term(&prep.s_list, &rho_hat, prep.x_design.ncols(), &final_fit);

    Ok(FittedGam {
        beta: final_fit.beta,
        rho: rho_hat,
        lambda: lambda_hat,
        scale,
        edf_total: edf,
        edf_per_term,
        n,
        n_iters: outer.iterations,
        converged: outer.converged && final_fit.converged,
        reml_value: outer.value,
        predictor: prep.predictor,
        vcov,
        link_kind,
    })
}

/// Default-Cholesky-backend shorthand for `fit_pirls_envelope`. 99% of
/// callers go through this; the explicit-S version exists for the
/// `gammon::fit::<_, _, _, LuSolver>(...)` opt-in.
#[allow(dead_code)]
pub(crate) fn fit_pirls_envelope_cholesky<L, K, V, M, P, SF>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    family: Family<L, K, V>,
    loss_marker: M,
    profile: P,
    scale_fn: SF,
    link_kind: LinkKind,
) -> Result<FittedGam>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    M: Loss,
    P: Profile<M>,
    SF: FnOnce(&GaussianInnerFit<CholeskySolver>, f64) -> f64,
{
    fit_pirls_envelope::<L, K, V, M, P, CholeskySolver, SF>(
        prep,
        x,
        y,
        prior_weights,
        family,
        loss_marker,
        profile,
        scale_fn,
        link_kind,
    )
}

/// Outer-Newton + shape-aware final fit + EDF for `ShapeAwareEnvelopeScore`.
/// Handles families with extra shape params jointly optimised with ρ
/// (TDist/scat, Tweedie, NegBin). Caller supplies:
/// - `family_base`: starting family (cloned per probe)
/// - `theta0`: initial θ = [ρ₀, shape₀…]
/// - `inner_builder`: how to instantiate the inner solver (PIRLS vs ocat).
/// - `profile`: dispersion convention (`FixedAtOneProfile` for scat /
///   NegBin / ocat, `OwnedByLossProfile` for Tweedie).
/// - `rebuild_final_family`: produces the final family with shape params
///   pulled from `outer.theta[1..]` (caller-specific because each family's
///   constructor takes different args).
/// - `scale_fn`: reports `FittedGam::scale` from the final fit + family.
///
/// Final fit reuses the score's `inner_builder.build(...)` (closes audit
/// §B4 / §79 — the final PIRLS must come from the same builder the score
/// used per probe).
pub(crate) fn fit_shape_aware<L, K, V, B, P, S, RF, SF>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    family_base: Family<L, K, V>,
    theta0: Array1<f64>,
    inner_builder: B,
    profile: P,
    rebuild_final_family: RF,
    scale_fn: SF,
    link_kind: LinkKind,
) -> Result<FittedGam>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S> + Clone,
    P: Profile<L>,
    S: LinearSolver,
    RF: FnOnce(&Array1<f64>) -> Family<L, K, V>,
    SF: FnOnce(&Family<L, K, V>, &GaussianInnerFit<S>, &Array1<f64>) -> f64,
{
    // 94b: shape-aware is single-smooth only — see ShapeAwareEnvelopeScore docs.
    if prep.s_list.len() != 1 {
        return Err(crate::error::GammonError::InvalidParameter(format!(
            "shape-aware families (TDist/scat, NegBin, Tweedie, Ocat) are restricted \
             to single-smooth fits in 94b; got {} terms",
            prep.s_list.len()
        )));
    }
    let prior = prior_weights.map(|w| w.to_owned());
    let n = x.nrows();

    let score = ShapeAwareEnvelopeScore::<L, K, V, B, P, S> {
        x_design: prep.x_design.clone(),
        y: y.to_owned(),
        prior_weights: prior.clone(),
        s_list: prep.s_list.clone(),
        family_base,
        rank_s_list: prep.rank_s_list.clone(),
        mp: prep.mp,
        log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: inner_builder.clone(),
        profile,
        _solver: PhantomData,
    };

    let outer_solver = NewtonWithHalving::new(NewtonOpts::default());
    let outer = outer_solver.minimize(&score, theta0)?;

    // Final fit: reuse the score's inner_builder so the final PIRLS sees
    // exactly the same solver settings as every outer probe (closes audit
    // §79 / §B4). Each family's constructor signature differs, so the
    // caller still supplies `rebuild_final_family`.
    let rho_hat = outer.theta[0];
    let family_final = rebuild_final_family(&outer.theta);
    let final_inner = score.inner_builder.build(
        family_final.clone(),
        prep.x_design.clone(),
        y.to_owned(),
        prior,
        prep.s_list.clone(),
        PirlsOpts::default(),
    );
    let final_fit: GaussianInnerFit<S> = final_inner.fit(&Array1::from_vec(vec![rho_hat]))?;

    let edf = compute_edf(&prep.x_design, &final_fit.working_weights, &final_fit);
    let scale = scale_fn(&family_final, &final_fit, &outer.theta);
    let vcov = compute_vcov(&final_fit, scale);
    let rho_vec = Array1::from_vec(vec![rho_hat]);
    let lambda_vec = Array1::from_vec(vec![rho_hat.exp()]);
    let edf_per_term =
        compute_edf_per_term(&prep.s_list, &rho_vec, prep.x_design.ncols(), &final_fit);

    Ok(FittedGam {
        beta: final_fit.beta,
        rho: rho_vec,
        lambda: lambda_vec,
        scale,
        edf_total: edf,
        edf_per_term,
        n,
        n_iters: outer.iterations,
        converged: outer.converged && final_fit.converged,
        reml_value: outer.value,
        predictor: prep.predictor,
        vcov,
        link_kind,
    })
}
