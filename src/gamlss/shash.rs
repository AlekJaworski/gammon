//! Sinh-ArcSinh (`shash`) density and its per-observation derivatives — the
//! likelihood core of the shash GAMLSS family.
//!
//! Parameterisation (matches mgcv `shash` and `gamrs._shash`):
//! params `(μ, τ, ε, φ)` with `σ = exp(τ)`, `δ = exp(φ)`,
//! `z = (y − μ)/(σδ)`, `dTasMe = δ·asinh(z) − ε`, and
//! ```text
//!   ℓ = −τ − ½log(2π) + log cosh(dTasMe) − ½log(1+z²) − ½sinh²(dTasMe) − phiPen·φ²
//! ```
//! Ported from mgcv `gamlss.r` (`shash`, lines 3491-3531); `phiPen` (default
//! 1e-3) is mgcv's tiny kurtosis regulariser. Derivatives here are in PARAM
//! space `(μ, τ, ε, φ)`; the link chain (μ identity, τ via `logeb`, ε/φ
//! identity) is applied separately when these feed the linear predictors.
//!
//! These functions are TDD-validated against two independent oracles: mgcv's
//! own `l0` (exact) and finite differences (`l1` = ∂`l0`, `l2` = ∂`l1`).

/// Sinh-ArcSinh log-density with mgcv's kurtosis penalty.
#[derive(Clone, Copy, Debug)]
pub struct ShashDensity {
    /// mgcv's `phiPen`: a `−phiPen·φ²` ridge on log-kurtosis (default 1e-3).
    pub phi_pen: f64,
}

impl Default for ShashDensity {
    fn default() -> Self {
        Self { phi_pen: 1e-3 }
    }
}

/// Lower-triangular packing order of the symmetric 4×4 param Hessian returned
/// by [`ShashDensity::l2`]: `[mm, mt, me, mp, tt, te, tp, ee, ep, pp]`.
pub const L2_INDEX: [(usize, usize); 10] = [
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3),
    (3, 3),
];

impl ShashDensity {
    /// Per-observation log-likelihood `ℓ(y; μ, τ, ε, φ)`.
    pub fn l0(&self, y: f64, mu: f64, tau: f64, eps: f64, phi: f64) -> f64 {
        let sig = tau.exp();
        let del = phi.exp();
        let z = (y - mu) / (sig * del);
        let d_tas_me = del * z.asinh() - eps;
        -tau - 0.5 * (2.0 * std::f64::consts::PI).ln() + d_tas_me.cosh().ln()
            - 0.5 * (z * z).ln_1p()
            - 0.5 * d_tas_me.sinh().powi(2)
            - self.phi_pen * phi * phi
    }

    /// Per-observation gradient `[∂ℓ/∂μ, ∂ℓ/∂τ, ∂ℓ/∂ε, ∂ℓ/∂φ]`.
    pub fn l1(&self, y: f64, mu: f64, tau: f64, eps: f64, phi: f64) -> [f64; 4] {
        let sig = tau.exp();
        let del = phi.exp();
        let z = (y - mu) / (sig * del);
        let asinh_z = z.asinh();
        let g = eps - del * asinh_z; // = −dTasMe
        let s_sp1 = (z * z + 1.0).sqrt();
        let zsd = z * sig * del; // = y − μ
        let de = g.tanh() - 0.5 * (2.0 * g).sinh();
        let dm = (1.0 / (del * sig * s_sp1)) * (del * de + z / s_sp1);
        let dt = zsd * dm - 1.0;
        let dp = dt + 1.0 - del * asinh_z * de - 2.0 * self.phi_pen * phi;
        [dm, dt, de, dp]
    }

    /// Per-observation Hessian, lower-triangular packed per [`L2_INDEX`]:
    /// `[Dmm, Dmt, Dme, Dmp, Dtt, Dte, Dtp, Dee, Dep, Dpp]` (observed, i.e.
    /// the data-dependent second derivative, matching mgcv's `l2`).
    pub fn l2(&self, y: f64, mu: f64, tau: f64, eps: f64, phi: f64) -> [f64; 10] {
        let sig = tau.exp();
        let del = phi.exp();
        let z = (y - mu) / (sig * del);
        let asinh_z = z.asinh();
        let g = eps - del * asinh_z; // = −dTasMe
        let s_sp1 = (z * z + 1.0).sqrt();
        let zsd = z * sig * del;
        let sech_g = 1.0 / g.cosh();
        let de = g.tanh() - 0.5 * (2.0 * g).sinh();
        let dm = (1.0 / (del * sig * s_sp1)) * (del * de + z / s_sp1);

        let dme = (sech_g * sech_g - (2.0 * g).cosh()) / (sig * s_sp1);
        let dte = zsd * dme;
        // mgcv `.ax2m1DivX2m2SQ(z, -1, 1)` = (z² − 1)/(z² + 1)²  (denom squared).
        let zz1 = z * z + 1.0;
        let ax = (z * z - 1.0) / (zz1 * zz1);
        let dmm = dme / (sig * s_sp1)
            + z * de / (sig * sig * del * s_sp1.powi(3))
            + ax / (del * sig * del * sig);
        let dmt = zsd * dmm - dm;
        let dee = -2.0 * g.cosh() * g.cosh() + sech_g * sech_g + 1.0;
        let dtt = zsd * dmt;
        let dep = dte - del * asinh_z * dee;
        let dmp = dmt + de / (sig * s_sp1) - del * asinh_z * dme;
        let dtp = zsd * dmp;
        let dpp =
            dtp - del * asinh_z * dep + del * (z / s_sp1 - asinh_z) * de - 2.0 * self.phi_pen;
        [dmm, dmt, dme, dmp, dtt, dte, dtp, dee, dep, dpp]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // (y, μ, τ, ε, φ, mgcv l0) — mgcv shash `ll`$l0 at the param point (phiPen=1e-3,
    // logeb b=1e-2), from scripts/r/gen_shash_deriv_fixture.R / fixture.
    const MGCV_L0: [(f64, f64, f64, f64, f64, f64); 5] = [
        (1.5, 0.0, -0.5, 0.0, 0.0, -3.4770055902),
        (2.3, 1.0, 0.0, 0.2, 0.0, -1.5650796503),
        (-0.5, 0.5, -1.0, -0.1, 0.3, -4.4213844739),
        (0.7, -0.3, 0.2, 0.0, -0.2, -1.5338829988),
        (3.0, 2.0, 0.3, 0.4, 0.1, -1.4063734522),
    ];

    #[test]
    fn l0_matches_mgcv() {
        let d = ShashDensity::default();
        for &(y, mu, tau, eps, phi, l0_ref) in &MGCV_L0 {
            let l0 = d.l0(y, mu, tau, eps, phi);
            assert!(
                (l0 - l0_ref).abs() < 1e-9,
                "l0({y},{mu},{tau},{eps},{phi}) = {l0:.10} vs mgcv {l0_ref:.10}"
            );
        }
    }

    // A spread of (y, params) for the FD checks — varied skew/kurtosis/scale,
    // y both above and below μ.
    fn fd_points() -> Vec<(f64, [f64; 4])> {
        vec![
            (1.5, [0.0, -0.5, 0.0, 0.0]),
            (2.3, [1.0, 0.0, 0.2, 0.0]),
            (-0.5, [0.5, -1.0, -0.1, 0.3]),
            (0.7, [-0.3, 0.2, 0.0, -0.2]),
            (3.0, [2.0, 0.3, 0.4, 0.1]),
            (-1.2, [-1.0, -0.7, -0.3, 0.2]),
            (0.4, [0.0, 0.5, 0.15, -0.15]),
        ]
    }

    fn l0_p(d: &ShashDensity, y: f64, p: [f64; 4]) -> f64 {
        d.l0(y, p[0], p[1], p[2], p[3])
    }
    fn l1_p(d: &ShashDensity, y: f64, p: [f64; 4]) -> [f64; 4] {
        d.l1(y, p[0], p[1], p[2], p[3])
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let d = ShashDensity::default();
        let h = 1e-6;
        for (y, p) in fd_points() {
            let g = l1_p(&d, y, p);
            for k in 0..4 {
                let (mut pp, mut pm) = (p, p);
                pp[k] += h;
                pm[k] -= h;
                let fd = (l0_p(&d, y, pp) - l0_p(&d, y, pm)) / (2.0 * h);
                assert!(
                    (g[k] - fd).abs() < 1e-6,
                    "∂ℓ/∂θ{k} at y={y}, p={p:?}: analytic {} vs FD {}",
                    g[k],
                    fd
                );
            }
        }
    }

    #[test]
    fn hessian_matches_finite_difference_and_is_symmetric() {
        let d = ShashDensity::default();
        let h = 1e-6;
        for (y, p) in fd_points() {
            let packed = d.l2(y, p[0], p[1], p[2], p[3]);
            // Unpack into a dense 4×4.
            let mut hess = [[0.0_f64; 4]; 4];
            for (idx, &(a, b)) in L2_INDEX.iter().enumerate() {
                hess[a][b] = packed[idx];
                hess[b][a] = packed[idx];
            }
            // l2[j][k] = ∂(l1[j])/∂θ_k — central FD of the gradient.
            for k in 0..4 {
                let (mut pp, mut pm) = (p, p);
                pp[k] += h;
                pm[k] -= h;
                let g_p = l1_p(&d, y, pp);
                let g_m = l1_p(&d, y, pm);
                for j in 0..4 {
                    let fd = (g_p[j] - g_m[j]) / (2.0 * h);
                    assert!(
                        (hess[j][k] - fd).abs() < 1e-5,
                        "∂²ℓ/∂θ{j}∂θ{k} at y={y}, p={p:?}: analytic {} vs FD {}",
                        hess[j][k],
                        fd
                    );
                }
            }
        }
    }
}
