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
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
];

/// Packing order of the symmetric 4×4×4 param third-derivative tensor returned
/// by [`ShashDensity::l3`]. Matches mgcv `gamlss.r:3570-3571`'s
/// `L3 <- cbind(Dmmm,Dmmt,Dmme,Dmmp,Dmtt,Dmte,Dmtp,Dmee,Dmep,Dmpp,
///              Dttt,Dtte,Dttp,Dtee,Dtep,Dtpp,Deee,Deep,Depp,Dppp)`
/// with μ=0, τ=1, ε=2, φ=3. Each triple is sorted non-decreasing; `l3[idx]` is
/// `∂³ℓ/∂θ_a∂θ_b∂θ_c` (fully symmetric, so only the 20 sorted triples are
/// stored).
pub const L3_INDEX: [(usize, usize, usize); 20] = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (0, 0, 3),
    (0, 1, 1),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 2),
    (0, 2, 3),
    (0, 3, 3),
    (1, 1, 1),
    (1, 1, 2),
    (1, 1, 3),
    (1, 2, 2),
    (1, 2, 3),
    (1, 3, 3),
    (2, 2, 2),
    (2, 2, 3),
    (2, 3, 3),
    (3, 3, 3),
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
        let dpp = dtp - del * asinh_z * dep + del * (z / s_sp1 - asinh_z) * de - 2.0 * self.phi_pen;
        [dmm, dmt, dme, dmp, dtt, dte, dtp, dee, dep, dpp]
    }

    /// Per-observation third derivatives in PARAM space `(μ, τ, ε, φ)`, packed
    /// per [`L3_INDEX`]: `[Dmmm, Dmmt, Dmme, Dmmp, Dmtt, Dmte, Dmtp, Dmee, Dmep,
    /// Dmpp, Dttt, Dtte, Dttp, Dtee, Dtep, Dtpp, Deee, Deep, Depp, Dppp]`.
    ///
    /// Ported verbatim from mgcv `gamlss.r:3545-3567` (the `deriv>1` `L3`
    /// block). The intermediates (`sig`, `del`, `z`, `asinhZ`, `g`, `sSp1`,
    /// `zsd`, `De`, `Dme`, `Dmm`, `Dee`, `Dmt`, …) are recomputed locally
    /// exactly as in [`Self::l2`]; the two `.ax2m1DivX2m2SQ` helpers are the
    /// rational forms `(a·z² + m1)/(z² + m2)²` (see [`ax2m1_div_x2m2_sq`]).
    pub fn l3(&self, y: f64, mu: f64, tau: f64, eps: f64, phi: f64) -> [f64; 20] {
        let sig = tau.exp();
        let del = phi.exp();
        let z = (y - mu) / (sig * del);
        let asinh_z = z.asinh();
        let g = eps - del * asinh_z; // = −dTasMe
        let s_sp1 = (z * z + 1.0).sqrt();
        let zsd = z * sig * del;
        let sech_g = 1.0 / g.cosh();

        // --- l1/l2 intermediates needed by the l3 recursions (mgcv 3511-3528).
        let de = g.tanh() - 0.5 * (2.0 * g).sinh();
        let dm = (1.0 / (del * sig * s_sp1)) * (del * de + z / s_sp1);
        let dme = (sech_g * sech_g - (2.0 * g).cosh()) / (sig * s_sp1);
        let dte = zsd * dme;
        // .ax2m1DivX2m2SQ(z, -1, 1) = (z² − 1)/(z² + 1)².
        let dmm = dme / (sig * s_sp1)
            + z * de / (sig * sig * del * s_sp1.powi(3))
            + ax2m1_div_x2m2_sq(z, -1.0, 1.0, 1.0) / (del * sig * del * sig);
        let dmt = zsd * dmm - dm;
        let dee = -2.0 * g.cosh() * g.cosh() + sech_g * sech_g + 1.0;
        let dep = dte - del * asinh_z * dee; // 2nd deriv ε,φ — reused by Dmpp/Dppp.
        let dmp = dmt + de / (sig * s_sp1) - del * asinh_z * dme;
        // De, Dme, Dee, Dmm, Dmt, Dmp, Dep feed the third-order recursions.

        // --- third derivatives (mgcv 3545-3567), in the exact source order.
        let deee = -2.0 * ((2.0 * g).sinh() + sech_g * sech_g * g.tanh());
        let dmee = deee / (sig * s_sp1);
        let dmme = dmee / (sig * s_sp1) + z * dee / (sig * sig * del * s_sp1.powi(3));
        // .ax2m1DivX2m2SQ(z, -1, 1, 2) = (2z² − 1)/(z² + 1)²,
        // .ax2m1DivX2m2SQ(z, -3, 1)    = (z² − 3)/(z² + 1)².
        let dmmm = 2.0 * z * dme / (sig * sig * del * s_sp1.powi(3))
            + dmme / (sig * s_sp1)
            + ax2m1_div_x2m2_sq(z, -1.0, 1.0, 2.0) * de / (sig.powi(3) * del * del * s_sp1)
            + 2.0 * (z / s_sp1) * ax2m1_div_x2m2_sq(z, -3.0, 1.0, 1.0)
                / ((sig * del).powi(3) * s_sp1);
        let dmmt = zsd * dmmm - 2.0 * dmm;
        let dtee = zsd * dmee;
        let dmte = zsd * dmme - dme;
        let dtte = zsd * dmte;
        let dmtt = zsd * dmmt - dmt;
        let dttt = zsd * dmtt;
        let dmep = dmte + dee / (sig * s_sp1) - del * asinh_z * dmee;
        let dtep = zsd * dmep;
        let deep = dtee - del * asinh_z * deee;
        let depp = dtep - del * asinh_z * deep + del * (z / s_sp1 - asinh_z) * dee;
        let dmmp = dmmt + 2.0 * dme / (sig * s_sp1) + z * de / (del * sig * sig * s_sp1.powi(3))
            - del * asinh_z * dmme;
        let dmtp = zsd * dmmp - dmp;
        let dttp = zsd * dmtp;
        let dmpp = dmtp + dep / (sig * s_sp1) + z * z * de / (sig * s_sp1.powi(3))
            - del * asinh_z * dmep
            + del * dme * (z / s_sp1 - asinh_z);
        let dtpp = zsd * dmpp;
        let dppp = dtpp - del * asinh_z * depp
            + del * (z / s_sp1 - asinh_z) * (2.0 * dep + de)
            + del * (z / s_sp1).powi(3) * de;

        [
            dmmm, dmmt, dmme, dmmp, dmtt, dmte, dmtp, dmee, dmep, dmpp, dttt, dtte, dttp, dtee,
            dtep, dtpp, deee, deep, depp, dppp,
        ]
    }

    // ---- Phase 2: link chain + η-space derivatives -----------------------
    //
    // The shash GAM solves for linear predictors `η = (η₁,η₂,η₃,η₄) = Xβ`; the
    // inner solver needs derivatives of ℓ w.r.t. η, not the params. mgcv's links
    // (linkinv = η → param) are:
    //   μ: identity  → μ = η₁
    //   τ: "logeb"   → τ = log(exp(η₂) + b), b = 1e-2  [ensures σ = exp(τ) ≥ b]
    //   ε: identity  → ε = η₃
    //   φ: identity  → φ = η₄
    // Only τ has a non-trivial link, so it is the only coordinate with a nonzero
    // `d²param/dη²` (which feeds the `j==k` term of the η-space Hessian).

    /// Apply mgcv's shash link inverses, mapping `η = (η₁,η₂,η₃,η₄)` to the
    /// params `(μ, τ, ε, φ)`. Only τ uses the `logeb` link with bound `b`.
    pub fn linkinv(eta: [f64; 4], b: f64) -> [f64; 4] {
        [eta[0], logeb_linkinv(eta[1], b), eta[2], eta[3]]
    }

    /// η-space gradient `[∂ℓ/∂η₁, ∂ℓ/∂η₂, ∂ℓ/∂η₃, ∂ℓ/∂η₄]`.
    ///
    /// Chain rule on the per-coordinate links: `G[k] = l1[k] · (dp_k/dη_k)`.
    pub fn l1_eta(&self, y: f64, eta: [f64; 4], b: f64) -> [f64; 4] {
        let [mu, tau, eps, phi] = Self::linkinv(eta, b);
        let l1 = self.l1(y, mu, tau, eps, phi);
        let dp = link_dparam(eta, b); // [dμ/dη₁, dτ/dη₂, dε/dη₃, dφ/dη₄]
        [l1[0] * dp[0], l1[1] * dp[1], l1[2] * dp[2], l1[3] * dp[3]]
    }

    /// η-space Hessian, lower-triangular packed per [`L2_INDEX`].
    ///
    /// Chain rule on the per-coordinate links:
    /// `H[j,k] = l2[j,k] · (dp_j/dη_j)(dp_k/dη_k) + (j==k ? l1[k]·(d²p_k/dη_k²) : 0)`.
    /// Only τ (index 1) has a nonzero `d²p/dη²`, so it is the sole diagonal
    /// correction; μ, ε, φ are identity links and contribute 0 there.
    pub fn l2_eta(&self, y: f64, eta: [f64; 4], b: f64) -> [f64; 10] {
        let [mu, tau, eps, phi] = Self::linkinv(eta, b);
        let l1 = self.l1(y, mu, tau, eps, phi);
        let l2 = self.l2(y, mu, tau, eps, phi);
        let dp = link_dparam(eta, b); // [dp_k/dη_k]
        let d2p = link_d2param(eta, b); // [d²p_k/dη_k²] (only index 1 nonzero)
        let mut out = [0.0_f64; 10];
        for (idx, &(j, k)) in L2_INDEX.iter().enumerate() {
            let mut h = l2[idx] * dp[j] * dp[k];
            if j == k {
                h += l1[k] * d2p[k];
            }
            out[idx] = h;
        }
        out
    }

    /// η-space third derivatives, packed per [`L3_INDEX`].
    ///
    /// Diagonal-link third-order chain rule (each param depends only on its own
    /// η). With `g_k = dp_k/dη_k` ([`link_dparam`]), `h_k = d²p_k/dη_k²`
    /// ([`link_d2param`], only τ nonzero), `j_k = d³p_k/dη_k³`
    /// ([`link_d3param`], only τ nonzero), and param-space `l1`, `l2`, `l3` at
    /// the linkinv params:
    /// ```text
    ///   l3_eta[a,b,c] = l3[a,b,c]·g_a·g_b·g_c
    ///                 + δ(a,c)·l2[a,b]·h_a·g_b
    ///                 + δ(b,c)·l2[a,b]·g_a·h_b
    ///                 + δ(a,b)·l2[a,c]·g_c·h_a
    ///                 + δ(a,b)·δ(a,c)·l1[a]·j_a
    /// ```
    /// (Drop the `h`/`j` terms and this is the pure `g_a g_b g_c` scaling; the
    /// curvature corrections only fire on repeated indices, and for shash only
    /// τ (index 1) has nonzero `h`/`j`.) The dense param `l2[·,·]`/`l3[·,·,·]`
    /// are unpacked from their symmetric storage; the result is repacked.
    pub fn l3_eta(&self, y: f64, eta: [f64; 4], b: f64) -> [f64; 20] {
        let [mu, tau, eps, phi] = Self::linkinv(eta, b);
        let l1 = self.l1(y, mu, tau, eps, phi);
        let l2_packed = self.l2(y, mu, tau, eps, phi);
        let l3_packed = self.l3(y, mu, tau, eps, phi);
        let g = link_dparam(eta, b); // [g_k] = dp_k/dη_k
        let hh = link_d2param(eta, b); // [h_k] = d²p_k/dη_k² (only k=1)
        let jj = link_d3param(eta, b); // [j_k] = d³p_k/dη_k³ (only k=1)

        // Unpack param l2 → dense 4×4.
        let mut l2 = [[0.0_f64; 4]; 4];
        for (idx, &(a, c)) in L2_INDEX.iter().enumerate() {
            l2[a][c] = l2_packed[idx];
            l2[c][a] = l2_packed[idx];
        }
        // Unpack param l3 → dense symmetric 4×4×4 (all permutations equal).
        let mut l3 = [[[0.0_f64; 4]; 4]; 4];
        for (idx, &(a, c, d)) in L3_INDEX.iter().enumerate() {
            let v = l3_packed[idx];
            for &(p, q, r) in &perms3(a, c, d) {
                l3[p][q][r] = v;
            }
        }

        let mut out = [0.0_f64; 20];
        for (idx, &(a, c, d)) in L3_INDEX.iter().enumerate() {
            // a,c,d are the η indices of this packed entry.
            let mut v = l3[a][c][d] * g[a] * g[c] * g[d];
            if a == d {
                v += l2[a][c] * hh[a] * g[c];
            }
            if c == d {
                v += l2[a][c] * g[a] * hh[c];
            }
            if a == c {
                v += l2[a][d] * g[d] * hh[a];
            }
            if a == c && a == d {
                v += l1[a] * jj[a];
            }
            out[idx] = v;
        }
        out
    }
}

/// All 6 ordered permutations of an index triple (with duplicates collapsed by
/// assignment) — used to fill the dense symmetric 4×4×4 third-derivative tensor
/// from its packed (sorted-triple) storage.
fn perms3(a: usize, b: usize, c: usize) -> [(usize, usize, usize); 6] {
    [
        (a, b, c),
        (a, c, b),
        (b, a, c),
        (b, c, a),
        (c, a, b),
        (c, b, a),
    ]
}

/// `logeb` link inverse `τ(η₂) = log(exp(η₂) + b)` (the scale link in shash;
/// `σ = exp(τ) ≥ b` by construction).
fn logeb_linkinv(eta2: f64, b: f64) -> f64 {
    (eta2.exp() + b).ln()
}

/// First derivative of the `logeb` link inverse: `dτ/dη₂ = exp(η₂)/(exp(η₂)+b)`.
fn logeb_dtau(eta2: f64, b: f64) -> f64 {
    let e = eta2.exp();
    e / (e + b)
}

/// Second derivative of the `logeb` link inverse:
/// `d²τ/dη₂² = b·exp(η₂)/(exp(η₂)+b)²`.
fn logeb_d2tau(eta2: f64, b: f64) -> f64 {
    let e = eta2.exp();
    let s = e + b;
    b * e / (s * s)
}

/// Per-coordinate link first derivatives `[dμ/dη₁, dτ/dη₂, dε/dη₃, dφ/dη₄]`.
/// μ, ε, φ are identity links (derivative 1); only τ uses `logeb`.
fn link_dparam(eta: [f64; 4], b: f64) -> [f64; 4] {
    [1.0, logeb_dtau(eta[1], b), 1.0, 1.0]
}

/// Per-coordinate link second derivatives `[d²μ/dη₁², …, d²φ/dη₄²]`. Identity
/// links have zero curvature, so only τ (`logeb`, index 1) is nonzero.
fn link_d2param(eta: [f64; 4], b: f64) -> [f64; 4] {
    [0.0, logeb_d2tau(eta[1], b), 0.0, 0.0]
}

/// Third derivative of the `logeb` link inverse. With `e = exp(η₂)`, `s = e + b`:
/// `d³τ/dη₂³ = b·e·(b − e)/s³` (the η-derivative of [`logeb_d2tau`]).
fn logeb_d3tau(eta2: f64, b: f64) -> f64 {
    let e = eta2.exp();
    let s = e + b;
    b * e * (b - e) / (s * s * s)
}

/// Per-coordinate link third derivatives `[d³μ/dη₁³, …, d³φ/dη₄³]`. Identity
/// links are zero; only τ (`logeb`, index 1) is nonzero ([`logeb_d3tau`]).
fn link_d3param(eta: [f64; 4], b: f64) -> [f64; 4] {
    [0.0, logeb_d3tau(eta[1], b), 0.0, 0.0]
}

/// mgcv's `.ax2m1DivX2m2SQ(x, m1, m2, a)` (gamlss.r:3454-3466): the rational
/// quantity `(a·x² + m1) / (x² + m2)²`. mgcv splits the evaluation into two
/// numerically-equivalent branches (a stable large-`|x|` form), but the value
/// is exactly this ratio; for the shash arguments (`a∈{1,2}`, `m1∈{−1,−3}`,
/// `m2=1`, |z| bounded) the plain form is accurate, so we evaluate it directly.
///
/// Used in [`ShashDensity::l2`]/[`ShashDensity::l3`] as
/// `(z²−1)/(z²+1)²` (a=1,m1=−1), `(2z²−1)/(z²+1)²` (a=2,m1=−1),
/// `(z²−3)/(z²+1)²` (a=1,m1=−3).
fn ax2m1_div_x2m2_sq(x: f64, m1: f64, m2: f64, a: f64) -> f64 {
    let x2 = x * x;
    let denom = x2 + m2;
    (a * x2 + m1) / (denom * denom)
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

    fn l2_p(d: &ShashDensity, y: f64, p: [f64; 4]) -> [f64; 10] {
        d.l2(y, p[0], p[1], p[2], p[3])
    }

    #[test]
    fn l3_matches_finite_difference() {
        // l3[a,b,c] = ∂(l2[a,b])/∂θ_c. Central FD of each packed l2 component
        // w.r.t. each of the four params, compared to the unpacked symmetric
        // 4×4×4 l3. The FD oracle is the authority for the analytic l3 port.
        let d = ShashDensity::default();
        let h = 1e-5;
        let mut max_err = 0.0_f64;
        for (y, p) in fd_points() {
            let l3_packed = d.l3(y, p[0], p[1], p[2], p[3]);
            // Dense symmetric 4×4×4.
            let mut l3 = [[[0.0_f64; 4]; 4]; 4];
            for (idx, &(a, b, c)) in L3_INDEX.iter().enumerate() {
                let v = l3_packed[idx];
                for &(x, yy, zz) in &perms3(a, b, c) {
                    l3[x][yy][zz] = v;
                }
            }
            // FD: differentiate each l2[pair] w.r.t. θ_c.
            for c in 0..4 {
                let (mut pp, mut pm) = (p, p);
                pp[c] += h;
                pm[c] -= h;
                let l2p = l2_p(&d, y, pp);
                let l2m = l2_p(&d, y, pm);
                for (idx, &(a, bb)) in L2_INDEX.iter().enumerate() {
                    let fd = (l2p[idx] - l2m[idx]) / (2.0 * h);
                    let err = (l3[a][bb][c] - fd).abs();
                    max_err = max_err.max(err);
                    assert!(
                        err < 1e-4,
                        "∂³ℓ/∂θ{a}∂θ{bb}∂θ{c} at y={y}, p={p:?}: analytic {} vs FD {} (err {err:.2e})",
                        l3[a][bb][c],
                        fd
                    );
                }
            }
            // Full symmetry of the 4×4×4: all permutations equal.
            for a in 0..4 {
                for bb in 0..4 {
                    for c in 0..4 {
                        for &(x, yy, zz) in &perms3(a, bb, c) {
                            assert!(
                                (l3[a][bb][c] - l3[x][yy][zz]).abs() < 1e-12,
                                "l3 asymmetry ({a},{bb},{c}) vs ({x},{yy},{zz})"
                            );
                        }
                    }
                }
            }
        }
        eprintln!("l3 (param) vs FD max err {max_err:.2e}");
    }

    // ---- Phase 2: link chain + η-space derivatives -----------------------

    /// mgcv's `logeb` bound `b` for the τ link (its default).
    const B_LOGEB: f64 = 1e-2;

    #[test]
    fn logeb_third_derivative_matches_finite_difference() {
        let h = 1e-6;
        for &eta2 in &[-2.0, -1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 2.0] {
            let d3 = logeb_d3tau(eta2, B_LOGEB);
            let fd = (logeb_d2tau(eta2 + h, B_LOGEB) - logeb_d2tau(eta2 - h, B_LOGEB)) / (2.0 * h);
            assert!(
                (d3 - fd).abs() < 1e-5,
                "d³τ/dη₂³ at η₂={eta2}: analytic {d3} vs FD {fd}"
            );
        }
    }

    // --- link function tests (analytic logeb derivatives vs finite diff) ---

    #[test]
    fn logeb_first_derivative_matches_finite_difference() {
        let h = 1e-6;
        for &eta2 in &[-2.0, -1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 2.0] {
            let d = logeb_dtau(eta2, B_LOGEB);
            let fd =
                (logeb_linkinv(eta2 + h, B_LOGEB) - logeb_linkinv(eta2 - h, B_LOGEB)) / (2.0 * h);
            assert!(
                (d - fd).abs() < 1e-6,
                "dτ/dη₂ at η₂={eta2}: analytic {d} vs FD {fd}"
            );
        }
    }

    #[test]
    fn logeb_second_derivative_matches_finite_difference() {
        let h = 1e-6;
        for &eta2 in &[-2.0, -1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 2.0] {
            let d2 = logeb_d2tau(eta2, B_LOGEB);
            let fd = (logeb_dtau(eta2 + h, B_LOGEB) - logeb_dtau(eta2 - h, B_LOGEB)) / (2.0 * h);
            assert!(
                (d2 - fd).abs() < 1e-5,
                "d²τ/dη₂² at η₂={eta2}: analytic {d2} vs FD {fd}"
            );
        }
    }

    // (y, η₁, η₂, η₃, η₄, mgcv l0) — mgcv shash `ll`$l0 at the RAW eta point
    // (phiPen=1e-3, logeb b=1e-2), from gen_shash_deriv_eta_fixture.R / fixture.
    // l0 already includes the logeb link + phiPen, so default().l0 at
    // linkinv(eta, b=1e-2) must reproduce it.
    const MGCV_L0_ETA: [(f64, f64, f64, f64, f64, f64); 4] = [
        (1.5, 0.0, -0.5, 0.0, 0.0, -3.3949604535092),
        (2.3, 1.0, 0.0, 0.2, 0.0, -1.5633689866257),
        (-0.5, 0.5, -1.0, -0.1, 0.3, -4.1366571933330),
        (0.7, -0.3, 0.2, 0.0, -0.2, -1.5361966478388),
    ];

    #[test]
    fn l0_at_eta_matches_mgcv() {
        let d = ShashDensity::default();
        for &(y, e1, e2, e3, e4, l0_ref) in &MGCV_L0_ETA {
            let [mu, tau, eps, phi] = ShashDensity::linkinv([e1, e2, e3, e4], B_LOGEB);
            let l0 = d.l0(y, mu, tau, eps, phi);
            assert!(
                (l0 - l0_ref).abs() < 1e-9,
                "l0@η({y},{e1},{e2},{e3},{e4}) = {l0:.13} vs mgcv {l0_ref:.13}"
            );
        }
    }

    // Spread of (y, η) for the FD checks — skew/kurtosis ±, scales, y above/below μ.
    fn eta_fd_points() -> Vec<(f64, [f64; 4])> {
        vec![
            (1.5, [0.0, -0.5, 0.0, 0.0]),
            (2.3, [1.0, 0.0, 0.2, 0.0]),
            (-0.5, [0.5, -1.0, -0.1, 0.3]),
            (0.7, [-0.3, 0.2, 0.0, -0.2]),
            (3.0, [2.0, 0.3, 0.4, 0.1]),
            (-1.2, [-1.0, -0.7, -0.3, 0.2]),
            (0.4, [0.0, 0.5, 0.15, -0.15]),
            (2.1, [1.5, -0.2, -0.25, 0.05]),
        ]
    }

    fn l0_eta(d: &ShashDensity, y: f64, eta: [f64; 4]) -> f64 {
        let [mu, tau, eps, phi] = ShashDensity::linkinv(eta, B_LOGEB);
        d.l0(y, mu, tau, eps, phi)
    }

    #[test]
    fn eta_gradient_matches_finite_difference() {
        let d = ShashDensity::default();
        let h = 1e-6;
        for (y, eta) in eta_fd_points() {
            let g = d.l1_eta(y, eta, B_LOGEB);
            for k in 0..4 {
                let (mut ep, mut em) = (eta, eta);
                ep[k] += h;
                em[k] -= h;
                let fd = (l0_eta(&d, y, ep) - l0_eta(&d, y, em)) / (2.0 * h);
                assert!(
                    (g[k] - fd).abs() < 1e-6,
                    "∂ℓ/∂η{k} at y={y}, η={eta:?}: analytic {} vs FD {}",
                    g[k],
                    fd
                );
            }
        }
    }

    #[test]
    fn eta_hessian_matches_finite_difference_and_is_symmetric() {
        let d = ShashDensity::default();
        let h = 1e-6;
        for (y, eta) in eta_fd_points() {
            let packed = d.l2_eta(y, eta, B_LOGEB);
            let mut hess = [[0.0_f64; 4]; 4];
            for (idx, &(a, b)) in L2_INDEX.iter().enumerate() {
                hess[a][b] = packed[idx];
                hess[b][a] = packed[idx];
            }
            for k in 0..4 {
                let (mut ep, mut em) = (eta, eta);
                ep[k] += h;
                em[k] -= h;
                let g_p = d.l1_eta(y, ep, B_LOGEB);
                let g_m = d.l1_eta(y, em, B_LOGEB);
                for j in 0..4 {
                    let fd = (g_p[j] - g_m[j]) / (2.0 * h);
                    assert!(
                        (hess[j][k] - fd).abs() < 1e-5,
                        "∂²ℓ/∂η{j}∂η{k} at y={y}, η={eta:?}: analytic {} vs FD {}",
                        hess[j][k],
                        fd
                    );
                }
            }
        }
    }

    #[test]
    fn eta_third_derivative_matches_finite_difference() {
        // l3_eta[a,b,c] = ∂(l2_eta[a,b])/∂η_c. Central FD of each packed
        // l2_eta component w.r.t. each η_c, compared to the unpacked symmetric
        // 4×4×4 l3_eta. This FD test is the authority for the η-space
        // chain-rule + the param-space l3 port together.
        let d = ShashDensity::default();
        let h = 1e-5;
        let mut max_err = 0.0_f64;
        for (y, eta) in eta_fd_points() {
            let l3_packed = d.l3_eta(y, eta, B_LOGEB);
            let mut l3 = [[[0.0_f64; 4]; 4]; 4];
            for (idx, &(a, b, c)) in L3_INDEX.iter().enumerate() {
                let v = l3_packed[idx];
                for &(x, yy, zz) in &perms3(a, b, c) {
                    l3[x][yy][zz] = v;
                }
            }
            for c in 0..4 {
                let (mut ep, mut em) = (eta, eta);
                ep[c] += h;
                em[c] -= h;
                let l2p = d.l2_eta(y, ep, B_LOGEB);
                let l2m = d.l2_eta(y, em, B_LOGEB);
                for (idx, &(a, bb)) in L2_INDEX.iter().enumerate() {
                    let fd = (l2p[idx] - l2m[idx]) / (2.0 * h);
                    let err = (l3[a][bb][c] - fd).abs();
                    max_err = max_err.max(err);
                    assert!(
                        err < 1e-4,
                        "∂³ℓ/∂η{a}∂η{bb}∂η{c} at y={y}, η={eta:?}: analytic {} vs FD {} (err {err:.2e})",
                        l3[a][bb][c],
                        fd
                    );
                }
            }
            // Full symmetry.
            for a in 0..4 {
                for bb in 0..4 {
                    for c in 0..4 {
                        for &(x, yy, zz) in &perms3(a, bb, c) {
                            assert!(
                                (l3[a][bb][c] - l3[x][yy][zz]).abs() < 1e-12,
                                "l3_eta asymmetry ({a},{bb},{c}) vs ({x},{yy},{zz})"
                            );
                        }
                    }
                }
            }
        }
        eprintln!("l3_eta vs FD max err {max_err:.2e}");
    }

    // --- η-space l1/l2 vs the mgcv-fixture finite-difference values ---------
    //
    // Embedded straight from tests/fixtures/shash_derivs_eta_mgcv.json. These
    // are mgcv's own central-FD of l0-at-η (fd_h=1e-4), so the analytic values
    // are compared to FD tolerances: ~1e-4 for l1, ~1e-3 for the second-FD l2.

    // (y, η₁..₄, l1[4]) from the fixture.
    const MGCV_L1_ETA: [(f64, [f64; 4], [f64; 4]); 4] = [
        (
            1.5,
            [0.0, -0.5, 0.0, 0.0],
            [3.9462255280, 4.8395476772, 5.4749160273, -2.9612197370],
        ),
        (
            2.3,
            [1.0, 0.0, 0.2, 0.0],
            [0.8930466406, 0.1593669700, 0.6804882557, 0.4324443319],
        ),
        (
            -0.5,
            [0.5, -1.0, -0.1, 0.3],
            [-11.2020920588, 9.9321094349, -8.6560445409, -5.4583237201],
        ),
        (
            0.7,
            [-0.3, 0.2, 0.0, -0.2],
            [0.7115936640, -0.2860642369, 0.3740611482, 0.4438304615],
        ),
    ];

    // (y, η₁..₄, l2[10]) from the fixture (lower-tri mm,mt,me,mp,tt,te,tp,ee,ep,pp).
    const MGCV_L2_ETA: [(f64, [f64; 4], [f64; 10]); 4] = [
        (
            1.5,
            [0.0, -0.5, 0.0, 0.0],
            [
                -2.63081663,
                -7.76443715,
                -7.82738701,
                8.17982712,
                -11.37925265,
                -11.55064273,
                12.07072816,
                -12.69415399,
                8.84940395,
                -5.90308433,
            ],
        ),
        (
            2.3,
            [1.0, 0.0, 0.2, 0.0],
            [
                -0.60787957,
                -1.66662378,
                -1.47728817,
                0.31162296,
                -2.14358142,
                -1.90146005,
                0.40109889,
                -2.43196816,
                0.68313894,
                -0.51939311,
            ],
        ),
        (
            -0.5,
            [0.5, -1.0, -0.1, 0.3],
            [
                -18.94629031,
                29.35055217,
                -23.01254414,
                -24.55116155,
                -28.31099621,
                22.40355266,
                23.90145328,
                -19.13786756,
                -13.82106343,
                -8.30274107,
            ],
        ),
        (
            0.7,
            [-0.3, 0.2, 0.0, -0.2],
            [
                -0.42720807,
                -1.12955377,
                -0.91944615,
                -0.26398329,
                -1.12270393,
                -0.91197950,
                -0.26183952,
                -1.59469642,
                0.22378650,
                -0.47890767,
            ],
        ),
    ];

    #[test]
    fn eta_gradient_matches_mgcv_fixture() {
        let d = ShashDensity::default();
        for &(y, eta, l1_ref) in &MGCV_L1_ETA {
            let g = d.l1_eta(y, eta, B_LOGEB);
            for k in 0..4 {
                assert!(
                    (g[k] - l1_ref[k]).abs() < 1e-4,
                    "η-l1[{k}] at y={y}, η={eta:?}: analytic {} vs mgcv-FD {}",
                    g[k],
                    l1_ref[k]
                );
            }
        }
    }

    #[test]
    fn eta_hessian_matches_mgcv_fixture() {
        let d = ShashDensity::default();
        for &(y, eta, l2_ref) in &MGCV_L2_ETA {
            let h = d.l2_eta(y, eta, B_LOGEB);
            for idx in 0..10 {
                assert!(
                    (h[idx] - l2_ref[idx]).abs() < 1e-3,
                    "η-l2[{idx}] at y={y}, η={eta:?}: analytic {} vs mgcv-FD {}",
                    h[idx],
                    l2_ref[idx]
                );
            }
        }
    }
}
