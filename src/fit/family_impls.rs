//! Per-family `FamilyFitWithSolver` impls.
//!
//! Each Loss impls `FamilyFitWithSolver<K, V, S>` exactly once for its
//! (Link, Variance) pair; the impl body is the validate-wire-drive
//! sequence that used to live in the deleted per-family
//! `fit_*_cr_with_solver` wrappers. The trait declaration and the
//! `gamrs::fit{,_with,_with_design,_with_solver}` public functions live in
//! `canonical.rs`.
//!
//! Impls consume a `PreparedDesign` built upstream by a
//! `DesignStrategy::prepare(x)` — basis choice is decoupled from family
//! dispatch.

use std::marker::PhantomData;

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::design::PreparedDesign;
use crate::error::{GamrsError, Result};
use crate::family::{
    bernoulli_logit, gamma_log, inverse_gaussian_log, negbin_log, ocat_identity, ocat_init_theta,
    poisson_log, quasibinomial_logit, quasipoisson_log, tdist_identity, tweedie_log, Bernoulli,
    BinomialVariance, ConstantVariance, ElfLoss, ElfVariance, Family, Gamma, GammaVariance,
    Gaussian, IdentityLink, InverseGaussian, InverseGaussianVariance, LogLink, LogitLink, NegBin,
    NegBinVariance, OcatLoss, OcatVariance, Poisson, PoissonVariance, QuasiBinomial, QuasiPoisson,
    TDist, TVariance, Tweedie, TweedieVariance,
};
use crate::inner::{GaussianInnerFit, LinearSolver, PirlsOpts};
use crate::outer::{NewtonOpts, NewtonWithHalving};
use crate::score::{
    FixedAtOneProfile, MgcvTwoSigmaProfile, OcatInnerBuilder, OwnedByLossProfile,
    PirlsInnerBuilder, ShapeAwareEnvelopeScore, ShapeInnerBuilder,
};
use crate::traits::{CoordsKind, InnerSolver, OuterSolver};

use super::canonical::FamilyFitWithSolver;
use super::driver::{fit_pirls_envelope, fit_shape_aware, make_pearson_scale_fn};
use super::gaussian::fit_gaussian_from_prep;
use super::quantile::fit_quantile_from_prep;
use super::{
    check_lengths, check_y_in_unit, check_y_nonneg, check_y_positive, compute_edf, compute_vcov,
    FittedGam, LinkKind,
};

// --- Gaussian: identity link + constant variance, closed-form inner ---
impl<S: LinearSolver> FamilyFitWithSolver<IdentityLink, ConstantVariance, S> for Gaussian {
    fn fit_from_prep_canonical(
        _family: Family<Self, IdentityLink, ConstantVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        fit_gaussian_from_prep::<S>(prep, x, y, prior_weights)
    }
}

// --- Bernoulli: logit + binomial variance, σ²≡1 ---
impl<S: LinearSolver> FamilyFitWithSolver<LogitLink, BinomialVariance, S> for Bernoulli {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogitLink, BinomialVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_in_unit(y, "Bernoulli")?;
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            bernoulli_logit(),
            Bernoulli,
            FixedAtOneProfile,
            |_, _| 1.0,
            LinkKind::Logit,
        )
    }
}

// --- Poisson: log link + Poisson variance, σ²≡1 ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, PoissonVariance, S> for Poisson {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogLink, PoissonVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_nonneg(y, "Poisson")?;
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            poisson_log(),
            Poisson,
            FixedAtOneProfile,
            |_, _| 1.0,
            LinkKind::Log,
        )
    }
}

// --- QuasiPoisson: log link + Poisson variance, profiled φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, PoissonVariance, S> for QuasiPoisson {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogLink, PoissonVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_nonneg(y, "QuasiPoisson")?;
        let n = x.nrows();
        let scale_fn = make_pearson_scale_fn::<S, _>(
            y.to_owned(),
            prior_weights.map(|w| w.to_owned()),
            n,
            1e-300,
            |mu_i| mu_i,
        );
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            quasipoisson_log(),
            QuasiPoisson,
            MgcvTwoSigmaProfile,
            scale_fn,
            LinkKind::Log,
        )
    }
}

// --- QuasiBinomial: logit + binomial variance, profiled φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogitLink, BinomialVariance, S> for QuasiBinomial {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogitLink, BinomialVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_in_unit(y, "QuasiBinomial")?;
        let n = x.nrows();
        let scale_fn = make_pearson_scale_fn::<S, _>(
            y.to_owned(),
            prior_weights.map(|w| w.to_owned()),
            n,
            1e-15,
            |mu_i| {
                let mu = mu_i.min(1.0 - 1e-15);
                mu * (1.0 - mu)
            },
        );
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            quasibinomial_logit(),
            QuasiBinomial,
            MgcvTwoSigmaProfile,
            scale_fn,
            LinkKind::Logit,
        )
    }
}

// --- Gamma: log link + μ² variance, profiled φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, GammaVariance, S> for Gamma {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogLink, GammaVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_positive(y, "Gamma")?;
        let n = x.nrows();
        let scale_fn = make_pearson_scale_fn::<S, _>(
            y.to_owned(),
            prior_weights.map(|w| w.to_owned()),
            n,
            1e-300,
            |mu_i| mu_i * mu_i,
        );
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            gamma_log(),
            Gamma,
            MgcvTwoSigmaProfile,
            scale_fn,
            LinkKind::Log,
        )
    }
}

// --- InverseGaussian: log link + μ³ variance, profiled φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, InverseGaussianVariance, S> for InverseGaussian {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogLink, InverseGaussianVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_positive(y, "InverseGaussian")?;
        let n = x.nrows();
        let scale_fn = make_pearson_scale_fn::<S, _>(
            y.to_owned(),
            prior_weights.map(|w| w.to_owned()),
            n,
            1e-300,
            |mu_i| mu_i * mu_i * mu_i,
        );
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            inverse_gaussian_log(),
            InverseGaussian,
            MgcvTwoSigmaProfile,
            scale_fn,
            LinkKind::Log,
        )
    }
}

// --- NegBin: log link + NegBin variance, shape-managed θ ---
// Initial θ is read off the family (`negbin_log(init_theta)` stamped it
// into `family.loss.theta`).
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, NegBinVariance, S> for NegBin {
    fn fit_from_prep_canonical(
        family: Family<Self, LogLink, NegBinVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        let init_theta = family.loss.theta;
        check_lengths(x, y, prior_weights)?;
        if !(init_theta > 0.0 && init_theta.is_finite()) {
            return Err(GamrsError::InvalidParameter(format!(
                "NegBin overdispersion init_theta must be a positive finite scalar (small θ → high variance, large θ → Poisson limit); got init_theta={init_theta}"
            )));
        }
        check_y_nonneg(y, "NegBin")?;

        let theta0 = Array1::from_vec(vec![0.0, init_theta.ln()]);

        fit_shape_aware::<_, _, _, _, _, S, _, _>(
            prep,
            x,
            y,
            prior_weights,
            negbin_log(init_theta),
            theta0,
            PirlsInnerBuilder,
            FixedAtOneProfile,
            |theta| {
                let mut f = negbin_log(init_theta);
                f.set_shape_params(&[theta[1]]);
                f
            },
            |family, _fit, _theta| family.loss.theta,
            LinkKind::Log,
        )
    }
}

// --- TDist (scat): identity link + T variance, shape-managed σ², ν ---
impl<S: LinearSolver> FamilyFitWithSolver<IdentityLink, TVariance, S> for TDist {
    fn fit_from_prep_canonical(
        family: Family<Self, IdentityLink, TVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        let init_nu = family.loss.nu;
        let init_sigma2 = family.loss.sigma2;
        check_lengths(x, y, prior_weights)?;
        if !(init_nu > 2.0 && init_nu.is_finite()) {
            return Err(GamrsError::InvalidParameter(format!(
                "scat/TDist degrees-of-freedom init_nu must be > 2 (finite variance); got init_nu={init_nu}"
            )));
        }
        if !(init_sigma2 > 0.0 && init_sigma2.is_finite()) {
            return Err(GamrsError::InvalidParameter(format!(
                "scat/TDist scale init_sigma2 must be a positive finite scalar; got init_sigma2={init_sigma2}"
            )));
        }

        let theta0 = Array1::from_vec(vec![0.0, init_sigma2.ln(), (init_nu - 2.0).ln()]);

        fit_shape_aware::<_, _, _, _, _, S, _, _>(
            prep,
            x,
            y,
            prior_weights,
            tdist_identity(init_nu, init_sigma2),
            theta0,
            PirlsInnerBuilder,
            FixedAtOneProfile,
            |theta| {
                let mut f = tdist_identity(init_nu, init_sigma2);
                f.set_shape_params(&[theta[1], theta[2]]);
                f
            },
            |_family, _fit, theta| theta[1].exp(),
            LinkKind::Identity,
        )
    }
}

// --- Tweedie: log link + μ^p variance, shape-managed p, φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, TweedieVariance, S> for Tweedie {
    fn fit_from_prep_canonical(
        family: Family<Self, LogLink, TweedieVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        let init_p = family.loss.p;
        let init_phi = family.loss.phi;
        check_lengths(x, y, prior_weights)?;
        if !(1.0 < init_p && init_p < 2.0) {
            return Err(GamrsError::InvalidParameter(format!(
                "Tweedie variance power must be in (1, 2) — strict interior \
                 (p=1 → Poisson, p=2 → Gamma — use those families directly); \
                 got init_p={init_p}"
            )));
        }
        if !(init_phi > 0.0 && init_phi.is_finite()) {
            return Err(GamrsError::InvalidParameter(format!(
                "Tweedie dispersion init_phi must be a positive finite scalar; got init_phi={init_phi}"
            )));
        }
        check_y_nonneg(y, "Tweedie")?;

        let p_t = ((init_p - 1.0) / (2.0 - init_p)).ln();
        let theta0 = Array1::from_vec(vec![0.0, init_phi.ln(), p_t]);

        fit_shape_aware::<_, _, _, _, _, S, _, _>(
            prep,
            x,
            y,
            prior_weights,
            tweedie_log(init_p, init_phi),
            theta0,
            PirlsInnerBuilder,
            OwnedByLossProfile,
            |theta| {
                let mut f = tweedie_log(init_p, init_phi);
                f.set_shape_params(&[theta[1], theta[2]]);
                f
            },
            // scale = φ̂ from the converged family.
            |family, _fit, _theta| family.loss.phi,
            LinkKind::Log,
        )
    }
}

// --- Ocat: identity link + Ocat variance, n_cats + thresholds on family ---
impl<S: LinearSolver> FamilyFitWithSolver<IdentityLink, OcatVariance, S> for OcatLoss {
    fn fit_from_prep_canonical(
        family: Family<Self, IdentityLink, OcatVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        let n_cats = family.loss.n_cats;
        let init_theta_opt: Option<Array1<f64>> = Some(family.loss.thresholds.clone());

        check_lengths(x, y, prior_weights)?;
        if n_cats < 3 {
            return Err(GamrsError::InvalidParameter(format!(
                "Ocat requires n_cats ≥ 3 (≤ 2 levels collapses to Bernoulli — use bernoulli_logit instead); got n_cats={n_cats}"
            )));
        }
        for (i, &yi) in y.iter().enumerate() {
            let r = yi.round() as i64;
            if r < 1 || (r as usize) > n_cats {
                return Err(GamrsError::InvalidParameter(format!(
                    "Ocat requires y to be an integer in 1..={n_cats}; got y={yi} at row {i}"
                )));
            }
        }

        let prior = prior_weights.map(|w| w.to_owned());
        let n = x.nrows();

        let theta0_shape: Array1<f64> =
            init_theta_opt.unwrap_or_else(|| ocat_init_theta(y, n_cats));
        if theta0_shape.len() != n_cats - 2 {
            return Err(GamrsError::InvalidParameter(format!(
                "Ocat init_theta length must equal n_cats - 2 = {} (log-gap thresholds between adjacent categories above the first); got {}",
                n_cats - 2,
                theta0_shape.len()
            )));
        }

        // 94b: shape-aware (ocat) is single-smooth only.
        if prep.s_list.len() != 1 {
            return Err(GamrsError::InvalidParameter(format!(
                "ocat is restricted to single-smooth fits in 94b; got {} terms",
                prep.s_list.len()
            )));
        }

        let family_base = ocat_identity(theta0_shape.clone(), n_cats);
        let score = ShapeAwareEnvelopeScore::<
            OcatLoss,
            IdentityLink,
            OcatVariance,
            OcatInnerBuilder,
            FixedAtOneProfile,
            S,
        > {
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
            inner_builder: OcatInnerBuilder,
            profile: FixedAtOneProfile,
            _solver: PhantomData,
        };

        // θ₀ = [log λ₀, θ₁, …, θ_{R-2}]
        let mut theta0 = Array1::<f64>::zeros(1 + theta0_shape.len());
        theta0[0] = 0.0;
        for (i, &t) in theta0_shape.iter().enumerate() {
            theta0[1 + i] = t;
        }
        let outer_solver = NewtonWithHalving::new(NewtonOpts::default());
        let outer = outer_solver.minimize(&score, theta0)?;

        let rho_hat = outer.theta[0];
        let theta_hat: Array1<f64> = outer.theta.slice(ndarray::s![1..]).to_owned();
        let family_final: Family<OcatLoss, IdentityLink, OcatVariance> =
            ocat_identity(theta_hat.clone(), n_cats);
        let final_inner = ShapeInnerBuilder::<OcatLoss, IdentityLink, OcatVariance, S>::build(
            &score.inner_builder,
            family_final.clone(),
            prep.x_design.clone(),
            y.to_owned(),
            prior,
            prep.s_list.clone(),
            PirlsOpts::default(),
        );
        let final_fit: GaussianInnerFit<S> = final_inner.fit(&Array1::from_vec(vec![rho_hat]))?;

        let edf = compute_edf(&prep.x_design, &final_fit.working_weights, &final_fit);
        let vcov = compute_vcov(&final_fit, 1.0);
        let rho_vec = Array1::from_vec(vec![rho_hat]);
        let lambda_vec = Array1::from_vec(vec![rho_hat.exp()]);
        let edf_per_term =
            super::compute_edf_per_term(&prep.s_list, &rho_vec, prep.x_design.ncols(), &final_fit);

        Ok(FittedGam {
            beta: final_fit.beta,
            rho: rho_vec,
            lambda: lambda_vec,
            scale: 1.0, // ocat has no dispersion
            edf_total: edf,
            edf_per_term,
            n,
            n_iters: outer.iterations,
            converged: outer.converged && final_fit.converged,
            reml_value: outer.value,
            predictor: prep.predictor,
            vcov,
            link_kind: LinkKind::Identity,
            shape_params: theta_hat,
        })
    }
}

// --- ELF (quantile): identity link + ElfVariance ---
// τ, σ, λ_elf all live on the Loss; warm-start picks defaults if
// init_sigma/init_lambda are <= 0.
impl<S: LinearSolver> FamilyFitWithSolver<IdentityLink, ElfVariance, S> for ElfLoss {
    fn fit_from_prep_canonical(
        family: Family<Self, IdentityLink, ElfVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        // tau validation lives in fit_quantile_from_prep (so the
        // helper carries actionable error messages for the row).
        fit_quantile_from_prep::<S>(
            prep,
            x,
            y,
            prior_weights,
            family.loss.tau,
            family.loss.sigma,
            family.loss.lambda,
        )
    }
}
