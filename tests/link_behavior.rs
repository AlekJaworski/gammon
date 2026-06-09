//! Behaviour tests for the non-identity link functions (Log, Logit, Inverse).
//!
//! Complements the round-trip / first-derivative checks already in
//! `family_tests.rs`. Focus here:
//!   - `InverseLink` (Gamma's canonical 1/μ link) end to end — it had **no**
//!     coverage before, despite feeding the Gamma IRLS + score paths.
//!   - the higher-order link derivatives the observed-Hessian / shape-gradient
//!     paths consume (`d2_link_dmu`, `d3_link_dmu`), checked against finite
//!     differences.
//!   - the chain-rule identity `dμ/dη · dη/dμ = 1` that every link must obey.
//!   - Logit's large-|η| numerical stability + endpoint clamping (the inputs
//!     PIRLS actually hits when η runs off during separation).

use gamrs::family::{IdentityLink, InverseLink, LogLink, LogitLink};
use gamrs::traits::Link;

fn central_fd<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn assert_rel(analytic: f64, fd: f64, tol: f64, ctx: &str) {
    let denom = fd.abs().max(1e-9);
    let rel = (analytic - fd).abs() / denom;
    assert!(
        rel < tol,
        "{ctx}: analytic={analytic:.6e} fd={fd:.6e} rel={rel:.2e} > {tol:.0e}"
    );
}

// ---------------------------------------------------------------------------
// InverseLink (Gamma) — the previously-untested link.
// ---------------------------------------------------------------------------

#[test]
fn inverse_link_round_trips() {
    let l = InverseLink;
    for &mu in &[1e-3, 0.05, 0.5, 1.0, 2.0, 17.0, 1.0e3] {
        let recovered = l.inverse_link(l.link(mu));
        assert!(
            (recovered - mu).abs() / mu < 1e-10,
            "μ={mu}: round-trip gave {recovered}"
        );
    }
}

#[test]
fn inverse_link_is_decreasing_and_positive_for_positive_mu() {
    // g(μ) = 1/μ is strictly decreasing; μ>0 ⟹ η>0.
    let l = InverseLink;
    let mut prev = f64::INFINITY;
    for &mu in &[0.1, 0.5, 1.0, 2.0, 10.0] {
        let eta = l.link(mu);
        assert!(eta > 0.0, "η must be >0 for μ={mu}; got {eta}");
        assert!(eta < prev, "1/μ must decrease in μ; μ={mu}");
        prev = eta;
    }
}

#[test]
fn inverse_link_derivatives_match_fd() {
    let l = InverseLink;
    // d_inverse_link is dμ/dη; FD against inverse_link. Stay away from η≈0
    // (1/η is singular there).
    for &eta in &[0.2_f64, 1.0, 5.0, -0.5, -3.0] {
        let h = 1e-5 * eta.abs().max(1e-3);
        let fd = central_fd(|e| l.inverse_link(e), eta, h);
        assert_rel(
            l.d_inverse_link(eta),
            fd,
            1e-5,
            &format!("d_inv_link η={eta}"),
        );
    }
    // d2/d3 link wrt μ feed the score/Hessian; FD the chain d_link→d2→d3.
    for &mu in &[0.2, 1.0, 5.0] {
        let h = 1e-5 * mu;
        let d2_fd = central_fd(|m| l.d_link_dmu(m), mu, h);
        assert_rel(l.d2_link_dmu(mu), d2_fd, 1e-5, &format!("inv d2 μ={mu}"));
        let d3_fd = central_fd(|m| l.d2_link_dmu(m), mu, h);
        assert_rel(l.d3_link_dmu(mu), d3_fd, 1e-4, &format!("inv d3 μ={mu}"));
    }
}

#[test]
fn inverse_link_finite_at_boundary_and_small_mu() {
    // The μ-floor keeps link / inverse_link / the FIRST derivative finite at
    // μ=0 — those are the values PIRLS feeds back. At exactly μ=0 the 1/μ link
    // is genuinely singular (and the cubed denominators in d2/d3 underflow the
    // floor to +inf), but for any realistic positive μ the higher derivatives
    // stay finite. Gamma μ is always > 0, so that's the contract that matters.
    let l = InverseLink;
    assert!(l.link(0.0).is_finite(), "link(0) must be floored finite");
    assert!(
        l.inverse_link(0.0).is_finite(),
        "inverse_link(0) floored finite"
    );
    assert!(
        l.d_link_dmu(0.0).is_finite(),
        "d_link_dmu(0) floored finite"
    );
    for &mu in &[1e-6, 1e-3, 0.1, 1.0, 100.0] {
        assert!(
            l.d2_link_dmu(mu).is_finite() && l.d3_link_dmu(mu).is_finite(),
            "d2/d3 must be finite at realistic μ={mu}"
        );
    }
}

// ---------------------------------------------------------------------------
// Higher-order derivatives for LogLink (first-deriv already tested elsewhere).
// ---------------------------------------------------------------------------

#[test]
fn log_link_higher_derivs_match_fd() {
    let l = LogLink;
    for &mu in &[0.2, 1.0, 5.0, 50.0] {
        let h = 1e-5 * mu;
        let d2_fd = central_fd(|m| l.d_link_dmu(m), mu, h);
        assert_rel(l.d2_link_dmu(mu), d2_fd, 1e-5, &format!("log d2 μ={mu}"));
        let d3_fd = central_fd(|m| l.d2_link_dmu(m), mu, h);
        assert_rel(l.d3_link_dmu(mu), d3_fd, 1e-4, &format!("log d3 μ={mu}"));
    }
}

// ---------------------------------------------------------------------------
// General chain-rule identity: every link obeys dμ/dη · dη/dμ = 1.
// ---------------------------------------------------------------------------

#[test]
fn link_chain_rule_identity_holds() {
    let links: Vec<(&str, Box<dyn Link>)> = vec![
        ("identity", Box::new(IdentityLink)),
        ("log", Box::new(LogLink)),
        ("logit", Box::new(LogitLink)),
        ("inverse", Box::new(InverseLink)),
    ];
    // μ ∈ (0,1): valid domain for all four links simultaneously.
    for (name, l) in &links {
        for &mu in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let eta = l.link(mu);
            let product = l.d_inverse_link(eta) * l.d_link_dmu(mu);
            assert!(
                (product - 1.0).abs() < 1e-8,
                "{name}: dμ/dη·dη/dμ = {product} ≠ 1 at μ={mu}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Logit stability — the inputs PIRLS hits when η runs off.
// ---------------------------------------------------------------------------

#[test]
fn logit_inverse_link_stable_and_in_unit_interval() {
    let l = LogitLink;
    let mut prev = -1.0;
    for &eta in &[-1.0e3, -40.0, -5.0, -1.0, 0.0, 1.0, 5.0, 40.0, 1.0e3] {
        let mu = l.inverse_link(eta);
        assert!(
            mu.is_finite() && (0.0..=1.0).contains(&mu),
            "η={eta}: μ={mu} out of [0,1] or non-finite"
        );
        assert!(
            mu >= prev,
            "inverse logit must be monotone increasing; η={eta}"
        );
        prev = mu;
    }
    assert!((l.inverse_link(0.0) - 0.5).abs() < 1e-12, "logit⁻¹(0)=0.5");
}

#[test]
fn logit_link_clamps_endpoints_finite() {
    // μ at/over the {0,1} boundary is clamped so logit stays finite.
    let l = LogitLink;
    for &mu in &[0.0, 1.0, -0.5, 1.5] {
        assert!(
            l.link(mu).is_finite(),
            "logit({mu}) must be finite (clamped)"
        );
    }
    assert!(l.link(0.0) < 0.0 && l.link(1.0) > 0.0, "endpoint signs");
}
