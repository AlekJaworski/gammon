//! Unit tests for `gammon::family` — Loss / Link / VarianceFn behaviour
//! per family. Lifted out of `src/family.rs` to keep that module within
//! the project's >700-LOC threshold (see `architecture-assumptions.md`
//! §G). The math is unchanged; this is a pure relocation modulo two
//! DRY helpers (`fd_oracle_d` for `analytic vs central FD` checks and
//! `shape_roundtrip` for `set/get_shape_params` round-trip).

use gammon::family::*;
use gammon::traits::{Link, Loss, VarianceFn};

// ─── DRY helpers ───────────────────────────────────────────────────────
//
// All the per-family `_d_loss_dmu_matches_fd` / `_d2_loss_dmu_matches_fd`
// tests share the same pattern: pick a Loss, run analytic at (y, μ),
// compare with the central-difference of `deviance_per_obs` (resp. the
// FD of `d_loss_dmu`) and assert a relative-error bound. Capture that
// in two helpers so each test reduces to a (loss, ys, μs) sweep.

/// FD oracle: `(F(μ+h) - F(μ-h)) / (2h)` rel-err vs `analytic`.
#[track_caller]
fn check_central_fd<F: Fn(f64) -> f64>(
    analytic: f64,
    f: F,
    mu: f64,
    h: f64,
    rel_tol: f64,
    ctx: &str,
) {
    let fd = (f(mu + h) - f(mu - h)) / (2.0 * h);
    let rel = (analytic - fd).abs() / (fd.abs() + 1.0);
    assert!(
        rel < rel_tol,
        "{ctx}: analytic={analytic} fd={fd} rel={rel}"
    );
}

/// `d_loss_dmu` FD oracle for Loss `L`.
#[track_caller]
fn fd_d_loss<L: Loss>(loss: &L, ys: &[f64], mus: &[f64], h: f64, rel_tol: f64) {
    for &y in ys {
        for &mu in mus {
            let analytic = loss.d_loss_dmu(y, mu);
            check_central_fd(
                analytic,
                |m| loss.deviance_per_obs(y, m),
                mu,
                h,
                rel_tol,
                &format!("y={y} μ={mu}"),
            );
        }
    }
}

/// `d2_loss_dmu` FD oracle for Loss `L`.
#[track_caller]
fn fd_d2_loss<L: Loss>(loss: &L, ys: &[f64], mus: &[f64], h: f64, rel_tol: f64) {
    for &y in ys {
        for &mu in mus {
            let analytic = loss.d2_loss_dmu(y, mu);
            check_central_fd(
                analytic,
                |m| loss.d_loss_dmu(y, m),
                mu,
                h,
                rel_tol,
                &format!("y={y} μ={mu}"),
            );
        }
    }
}

// ─── Gaussian / Identity / Constant ────────────────────────────────────

#[test]
fn gaussian_deviance_is_squared_residual() {
    let g = Gaussian;
    assert_eq!(g.deviance_per_obs(3.0, 1.0), 4.0);
    assert_eq!(g.deviance_per_obs(0.0, 0.0), 0.0);
}

#[test]
fn identity_link_round_trips() {
    let l = IdentityLink;
    for &v in &[-1.5, 0.0, 0.7, 100.0] {
        assert_eq!(l.inverse_link(l.link(v)), v);
        assert_eq!(l.d_inverse_link(v), 1.0);
    }
}

// ─── Bernoulli / Logit / Binomial ──────────────────────────────────────

#[test]
fn bernoulli_deviance_is_zero_at_y_eq_mu() {
    let b = Bernoulli;
    for &y in &[0.0, 1.0] {
        assert!(b.deviance_per_obs(y, y).abs() < 1e-10);
    }
}

#[test]
fn bernoulli_d_loss_dmu_matches_fd() {
    fd_d_loss(&Bernoulli, &[0.0, 1.0], &[0.1, 0.3, 0.5, 0.7, 0.9], 1e-6, 1e-5);
}

#[test]
fn bernoulli_d2_loss_dmu_matches_fd() {
    fd_d2_loss(&Bernoulli, &[0.0, 1.0], &[0.2, 0.5, 0.8], 1e-4, 1e-3);
}

#[test]
fn logit_link_round_trips() {
    let l = LogitLink;
    for &mu in &[0.01, 0.3, 0.5, 0.7, 0.99] {
        let recovered = l.inverse_link(l.link(mu));
        assert!((recovered - mu).abs() < 1e-12, "mu={mu} recovered={recovered}");
    }
}

#[test]
fn logit_d_inverse_link_matches_fd() {
    let l = LogitLink;
    let h = 1e-6;
    for &eta in &[-3.0, -0.5, 0.0, 0.5, 3.0] {
        let analytic = l.d_inverse_link(eta);
        let fd = (l.inverse_link(eta + h) - l.inverse_link(eta - h)) / (2.0 * h);
        assert!(
            (analytic - fd).abs() < 1e-7,
            "eta={eta}: analytic={analytic} fd={fd}"
        );
    }
}

#[test]
fn logit_d_link_dmu_is_reciprocal_of_d_inverse_link() {
    let l = LogitLink;
    for &mu in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        let eta = l.link(mu);
        let dn_dmu = l.d_link_dmu(mu);
        let dmu_deta = l.d_inverse_link(eta);
        assert!((dn_dmu * dmu_deta - 1.0).abs() < 1e-10);
    }
}

#[test]
fn binomial_variance_is_mu_one_minus_mu() {
    let v = BinomialVariance;
    for &mu in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        assert!((v.variance(mu) - mu * (1.0 - mu)).abs() < 1e-12);
    }
}

// ─── TDist / scat ──────────────────────────────────────────────────────

#[test]
fn tdist_deviance_zero_at_y_eq_mu() {
    let t = TDist { nu: 5.0, sigma2: 0.25 };
    assert!(t.deviance_per_obs(0.5, 0.5).abs() < 1e-12);
}

#[test]
fn tdist_d_loss_dmu_matches_fd() {
    for &nu in &[3.0, 5.0, 10.0] {
        for &sigma2 in &[0.1, 1.0, 4.0] {
            let t = TDist { nu, sigma2 };
            for &mu in &[-0.5, 0.0, 0.5, 1.5] {
                let ys = [mu - 1.0, mu - 0.3, mu, mu + 0.3, mu + 1.0];
                fd_d_loss(&t, &ys, &[mu], 1e-6, 1e-5);
            }
        }
    }
}

#[test]
fn tdist_d2_loss_dmu_matches_fd() {
    for &nu in &[3.0, 5.0, 10.0] {
        for &sigma2 in &[0.5, 2.0] {
            let t = TDist { nu, sigma2 };
            for &mu in &[0.0, 1.0] {
                let ys = [mu - 1.5, mu - 0.5, mu, mu + 0.5, mu + 1.5];
                fd_d2_loss(&t, &ys, &[mu], 1e-4, 1e-3);
            }
        }
    }
}

#[test]
fn tdist_shape_params_roundtrip() {
    let original = TDist { nu: 4.5, sigma2: 0.16 };
    let params = original.get_shape_params();
    assert_eq!(params.len(), 2);
    // params = [log(0.16), log(4.5 - 2)]
    assert!((params[0] - 0.16_f64.ln()).abs() < 1e-12);
    assert!((params[1] - 2.5_f64.ln()).abs() < 1e-12);

    let mut restored = TDist { nu: 99.0, sigma2: 99.0 };
    restored.set_shape_params(&params);
    assert!((restored.nu - 4.5).abs() < 1e-12);
    assert!((restored.sigma2 - 0.16).abs() < 1e-12);
}

#[test]
fn family_shape_params_sync_loss_and_variance() {
    // Setting shape params on the Family aggregator must propagate
    // σ² into both the Loss and the Variance (the dual-storage
    // problem). Without this, PIRLS would read inconsistent σ²
    // from `variance` vs the score body reading `loss.sigma2`.
    let mut fam = tdist_identity(4.0, 1.0);
    let new_params = vec![(0.25_f64).ln(), (3.0_f64).ln()]; // σ²=0.25, ν=5
    fam.set_shape_params(&new_params);
    assert!((fam.loss.sigma2 - 0.25).abs() < 1e-12);
    assert!((fam.loss.nu - 5.0).abs() < 1e-12);
    assert!((fam.variance.sigma2 - 0.25).abs() < 1e-12);
}

#[test]
fn tdist_d2_is_negative_for_outliers() {
    // Robust-regression property: d²L/dμ² should turn NEGATIVE when
    // |y - μ| > √(ν·σ²), i.e. outliers reduce the loss curvature.
    let t = TDist { nu: 4.0, sigma2: 1.0 };
    let threshold = (t.nu * t.sigma2).sqrt(); // = 2.0
    assert!(t.d2_loss_dmu(0.0, 0.0) > 0.0);
    assert!(t.d2_loss_dmu(1.0, 0.0) > 0.0);
    assert!(t.d2_loss_dmu(threshold + 0.5, 0.0) < 0.0);
    assert!(t.d2_loss_dmu(-(threshold + 0.5), 0.0) < 0.0);
}

// ─── Poisson / Log / PoissonVariance ───────────────────────────────────

#[test]
fn poisson_deviance_zero_at_y_eq_mu() {
    let p = Poisson;
    for &y in &[0.0, 1.0, 5.0, 100.0] {
        let mu = if y == 0.0 { 1e-300 } else { y };
        let d = p.deviance_per_obs(y, mu);
        assert!(d.abs() < 1e-6, "y={y} μ={mu}: D = {d}");
    }
}

#[test]
fn poisson_d_loss_dmu_matches_fd() {
    fd_d_loss(&Poisson, &[0.0, 1.0, 5.0, 10.0], &[0.5, 1.0, 3.0, 8.0, 20.0], 1e-6, 1e-5);
}

#[test]
fn poisson_d2_loss_dmu_matches_fd() {
    fd_d2_loss(&Poisson, &[1.0, 5.0, 10.0], &[0.5, 1.0, 3.0, 8.0], 1e-4, 1e-3);
}

#[test]
fn log_link_round_trips() {
    let l = LogLink;
    for &mu in &[0.1, 1.0, 5.0, 100.0] {
        let recovered = l.inverse_link(l.link(mu));
        assert!((recovered - mu).abs() / mu < 1e-12, "μ={mu} recovered={recovered}");
    }
}

#[test]
fn log_d_link_dmu_is_reciprocal_of_d_inverse_link() {
    let l = LogLink;
    for &mu in &[0.1, 0.5, 1.0, 3.0, 10.0] {
        let eta = l.link(mu);
        let dn_dmu = l.d_link_dmu(mu);
        let dmu_deta = l.d_inverse_link(eta);
        assert!((dn_dmu * dmu_deta - 1.0).abs() < 1e-10);
    }
}

#[test]
fn poisson_variance_equals_mu() {
    let v = PoissonVariance;
    for &mu in &[0.1, 1.0, 5.0] {
        assert!((v.variance(mu) - mu).abs() < 1e-12);
    }
}

// ─── NegBin ────────────────────────────────────────────────────────────

#[test]
fn negbin_deviance_zero_at_y_eq_mu_for_positive_y() {
    let n = NegBin { theta: 2.0 };
    for &y in &[1.0, 3.0, 10.0] {
        let d = n.deviance_per_obs(y, y);
        assert!(d.abs() < 1e-10, "y={y}: D = {d}");
    }
}

#[test]
fn negbin_d_loss_dmu_matches_fd() {
    for &theta in &[0.5, 2.0, 10.0] {
        fd_d_loss(&NegBin { theta }, &[0.0, 1.0, 5.0, 20.0], &[0.5, 1.0, 5.0, 20.0], 1e-6, 1e-5);
    }
}

#[test]
fn negbin_d2_loss_dmu_matches_fd() {
    for &theta in &[1.0, 5.0] {
        fd_d2_loss(&NegBin { theta }, &[0.0, 2.0, 10.0], &[1.0, 3.0, 8.0], 1e-4, 1e-3);
    }
}

#[test]
fn negbin_variance_is_mu_plus_mu_sq_over_theta() {
    let v = NegBinVariance { theta: 4.0 };
    for &mu in &[1.0, 3.0, 10.0] {
        let expected = mu + mu * mu / 4.0;
        assert!((v.variance(mu) - expected).abs() < 1e-12);
    }
}

#[test]
fn negbin_recovers_poisson_at_theta_infinity() {
    let n_big = NegBin { theta: 1e10 };
    let p = Poisson;
    for &y in &[0.0, 1.0, 5.0] {
        for &mu in &[0.5, 1.0, 5.0] {
            let dn = n_big.deviance_per_obs(y, mu);
            let dp = p.deviance_per_obs(y, mu);
            let rel = (dn - dp).abs() / (dp.abs() + 1.0);
            assert!(rel < 1e-3, "θ→∞ y={y} μ={mu}: NB={dn} Poi={dp}");
        }
    }
}

#[test]
fn negbin_shape_params_roundtrip() {
    let n = NegBin { theta: 3.7 };
    let p = n.get_shape_params();
    assert_eq!(p.len(), 1);
    assert!((p[0] - 3.7_f64.ln()).abs() < 1e-12);

    let mut n2 = NegBin { theta: 99.0 };
    n2.set_shape_params(&p);
    assert!((n2.theta - 3.7).abs() < 1e-12);
}

// ─── Gamma / InverseGaussian ───────────────────────────────────────────

#[test]
fn gamma_deviance_zero_at_y_eq_mu() {
    let g = Gamma;
    for &y in &[0.1, 1.0, 5.0] {
        assert!(g.deviance_per_obs(y, y).abs() < 1e-10);
    }
}

#[test]
fn gamma_d_loss_dmu_matches_fd() {
    fd_d_loss(&Gamma, &[0.1, 1.0, 5.0, 20.0], &[0.5, 1.0, 5.0, 20.0], 1e-6, 1e-5);
}

#[test]
fn gamma_variance_is_mu_squared() {
    let v = GammaVariance;
    for &mu in &[0.5, 1.0, 5.0] {
        assert!((v.variance(mu) - mu * mu).abs() < 1e-12);
    }
}

#[test]
fn invgauss_deviance_zero_at_y_eq_mu() {
    let ig = InverseGaussian;
    for &y in &[0.5, 1.0, 5.0] {
        assert!(ig.deviance_per_obs(y, y).abs() < 1e-12);
    }
}

#[test]
fn invgauss_d_loss_dmu_matches_fd() {
    fd_d_loss(&InverseGaussian, &[0.5, 1.0, 5.0], &[0.5, 1.0, 5.0], 1e-6, 1e-5);
}

#[test]
fn invgauss_variance_is_mu_cubed() {
    let v = InverseGaussianVariance;
    for &mu in &[0.5, 1.0, 3.0] {
        assert!((v.variance(mu) - mu * mu * mu).abs() < 1e-12);
    }
}

// ─── Tweedie ───────────────────────────────────────────────────────────

#[test]
fn tweedie_d_loss_dmu_matches_fd() {
    for &p in &[1.3, 1.5, 1.7] {
        fd_d_loss(&Tweedie { p, phi: 1.0 }, &[0.0, 1.0, 5.0], &[0.5, 1.0, 3.0], 1e-6, 1e-5);
    }
}

#[test]
fn tweedie_variance_is_mu_pow_p() {
    let v = TweedieVariance { p: 1.5 };
    for &mu in &[0.5_f64, 1.0, 4.0] {
        let expected = mu.powf(1.5);
        assert!((v.variance(mu) - expected).abs() < 1e-12);
    }
}

#[test]
fn tweedie_shape_params_roundtrip() {
    let t = Tweedie { p: 1.6, phi: 0.5 };
    let params = t.get_shape_params();
    let mut t2 = Tweedie { p: 99.0, phi: 99.0 };
    t2.set_shape_params(&params);
    assert!((t2.p - 1.6).abs() < 1e-10, "p got {}", t2.p);
    assert!((t2.phi - 0.5).abs() < 1e-10, "phi got {}", t2.phi);
}

// ─── Ocat ──────────────────────────────────────────────────────────────

#[test]
fn ocat_deviance_is_nonnegative_and_finite() {
    let theta = ndarray::Array1::from_vec(vec![0.5_f64, 0.5]);
    let ocat = OcatLoss::new(theta, 4);
    for &y in &[1.0_f64, 2.0, 3.0, 4.0] {
        for &mu in &[-3.0_f64, -1.0, 0.0, 1.0, 3.0] {
            let d = ocat.deviance_per_obs(y, mu);
            assert!(d.is_finite() && d >= 0.0, "y={y} μ={mu}: D = {d}");
        }
    }
}

#[test]
fn ocat_d_loss_dmu_matches_fd() {
    let theta = ndarray::Array1::from_vec(vec![0.4_f64, 0.6]);
    let ocat = OcatLoss::new(theta, 4);
    fd_d_loss(&ocat, &[1.0, 2.0, 3.0, 4.0], &[-0.3, 0.1, 0.7, 1.2], 1e-6, 1e-5);
}

#[test]
fn ocat_d2_loss_dmu_matches_fd_of_dmu() {
    let theta = ndarray::Array1::from_vec(vec![0.4_f64, 0.6]);
    let ocat = OcatLoss::new(theta, 4);
    fd_d2_loss(&ocat, &[1.0, 2.0, 3.0, 4.0], &[-0.3, 0.1, 0.7, 1.2], 1e-4, 1e-3);
}

#[test]
fn ocat_shape_params_roundtrip() {
    let theta = ndarray::Array1::from_vec(vec![0.4_f64, -0.2, 0.7]);
    let ocat = OcatLoss::new(theta.clone(), 5);
    assert_eq!(ocat.n_shape_params(), 3);
    let got = ocat.get_shape_params();
    for (a, b) in got.iter().zip(theta.iter()) {
        assert!((a - b).abs() < 1e-14);
    }
    let mut ocat2 = OcatLoss::new(ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0]), 5);
    ocat2.set_shape_params(&[0.4, -0.2, 0.7]);
    assert!((ocat2.thresholds[0] - 0.4).abs() < 1e-14);
    assert!((ocat2.thresholds[1] - (-0.2)).abs() < 1e-14);
    assert!((ocat2.thresholds[2] - 0.7).abs() < 1e-14);
}

#[test]
fn ocat_init_theta_returns_correct_length() {
    let y = ndarray::Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 4.0]);
    let theta = ocat_init_theta(y.view(), 4);
    assert_eq!(theta.len(), 2);
    for &t in theta.iter() {
        assert!(t.is_finite(), "θ = {t}");
    }
}
