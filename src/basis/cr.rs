//! Cubic-regression-spline basis (Wood 2017 §5.3.1 / mgcv `bs="cr"`).
//!
//! Cardinal natural cubic spline through `k` knots: basis function `j` is
//! the unique natural cubic spline with value 1 at knot `j` and 0 at every
//! other knot. Penalty is the integrated squared second derivative,
//! `S = D' B⁻¹ D` per Wood (2006) §4.1.2 / mgcv `getFS`.
//!
//! Phase 0 reuses the same construction recipe as `src/basis.rs` and
//! `src/penalty.rs` in the v0.x crate. Byte-equivalence with v0.x is the
//! parity bar.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{GammonError, Result};
use crate::traits::Basis;

pub struct CrSpline {
    knots: Array1<f64>,
    k: usize,
}

impl CrSpline {
    pub fn new(knots: Array1<f64>) -> Result<Self> {
        let k = knots.len();
        if k < 3 {
            return Err(GammonError::InvalidParameter(format!(
                "CR spline needs k ≥ 3 (got {})",
                k
            )));
        }
        // Knots must be strictly increasing.
        for w in knots.as_slice().unwrap().windows(2) {
            if !(w[1] > w[0]) {
                return Err(GammonError::InvalidParameter(
                    "CR spline knots must be strictly increasing".into(),
                ));
            }
        }
        Ok(Self { knots, k })
    }

    /// Quantile knots over unique values of `x` — mirrors mgcv's
    /// `smooth.construct.cr.smooth.spec` (and v0.x's `with_quantile_knots`).
    /// `x` is a 1-D view (the univariate input).
    pub fn with_quantile_knots(x: ArrayView1<f64>, k: usize) -> Result<Self> {
        let mut sorted: Vec<f64> = x.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut unique: Vec<f64> = Vec::with_capacity(sorted.len());
        for &v in &sorted {
            if unique.last().is_none_or(|&last| (v - last).abs() > 1e-12) {
                unique.push(v);
            }
        }
        if unique.len() < k {
            let lo = *unique.first().unwrap_or(&0.0);
            let hi = *unique.last().unwrap_or(&1.0);
            return Self::new(Array1::linspace(lo, hi, k));
        }
        let n = unique.len();
        let mut knots = Vec::with_capacity(k);
        for i in 0..k {
            let q = i as f64 / (k - 1) as f64;
            let pos = q * (n - 1) as f64;
            let idx = pos.floor() as usize;
            let knot = if idx >= n - 1 {
                unique[n - 1]
            } else {
                let frac = pos - idx as f64;
                unique[idx] * (1.0 - frac) + unique[idx + 1] * frac
            };
            knots.push(knot);
        }
        Self::new(Array1::from_vec(knots))
    }

    pub fn knots(&self) -> ArrayView1<'_, f64> {
        self.knots.view()
    }

    /// Standard tridiagonal solver for the natural-cubic-spline second
    /// derivatives. Mirrors `src/basis.rs::solve_tridiagonal` exactly.
    fn solve_natural_tri(&self, h: &[f64], alpha: &[f64]) -> Vec<f64> {
        let n = self.k - 1;
        let mut c = vec![0.0; n + 1];
        let mut l = vec![0.0; n + 1];
        let mut mu = vec![0.0; n + 1];
        let mut z = vec![0.0; n + 1];
        l[0] = 1.0;
        for i in 1..n {
            l[i] = 2.0 * (h[i - 1] + h[i]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        l[n] = 1.0;
        for j in (0..n).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
        }
        c
    }
}

/// Pre-computed coefficients of the cardinal natural-cubic-spline basis.
/// Used internally by both `evaluate` and `d1` so they share one
/// construction. Each `*_co[i*k + j]` slot holds the `b/c/d` coefficient
/// of basis function `j` on interval `[knot_i, knot_{i+1}]`.
struct CrCoeffs {
    #[allow(dead_code)] // kept for future blanket d2 support
    h: Vec<f64>,
    vals_at: Vec<f64>,
    b_co: Vec<f64>,
    c_co: Vec<f64>,
    d_co: Vec<f64>,
    vals_0: Vec<f64>,
    vals_last: Vec<f64>,
    left_slope: Vec<f64>,
    right_slope: Vec<f64>,
}

impl CrSpline {
    fn build_coeffs(&self) -> CrCoeffs {
        let k = self.k;
        let num_int = k - 1;
        let h: Vec<f64> = (0..num_int).map(|i| self.knots[i + 1] - self.knots[i]).collect();

        let mut vals_at = vec![0.0; k * k];
        let mut b_co = vec![0.0; num_int * k];
        let mut c_co = vec![0.0; k * k];
        let mut d_co = vec![0.0; num_int * k];
        let mut vals_0 = vec![0.0; k];
        let mut vals_last = vec![0.0; k];
        let mut left_slope = vec![0.0; k];
        let mut right_slope = vec![0.0; k];

        for j in 0..k {
            let mut values = vec![0.0; k];
            values[j] = 1.0;
            let mut alpha = vec![0.0; k];
            for i in 1..num_int {
                alpha[i] = (3.0 / h[i]) * (values[i + 1] - values[i])
                    - (3.0 / h[i - 1]) * (values[i] - values[i - 1]);
            }
            let c = self.solve_natural_tri(&h, &alpha);
            for i in 0..num_int {
                let b_i =
                    (values[i + 1] - values[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
                let d_i = (c[i + 1] - c[i]) / (3.0 * h[i]);
                vals_at[i * k + j] = values[i];
                b_co[i * k + j] = b_i;
                c_co[i * k + j] = c[i];
                d_co[i * k + j] = d_i;
            }
            vals_at[num_int * k + j] = values[num_int];
            vals_0[j] = values[0];
            vals_last[j] = values[num_int];
            left_slope[j] = b_co[j];
            let last_i = num_int - 1;
            let hn = h[last_i];
            right_slope[j] = b_co[last_i * k + j]
                + 2.0 * c_co[last_i * k + j] * hn
                + 3.0 * d_co[last_i * k + j] * hn * hn;
        }

        CrCoeffs {
            h,
            vals_at,
            b_co,
            c_co,
            d_co,
            vals_0,
            vals_last,
            left_slope,
            right_slope,
        }
    }
}

impl Basis for CrSpline {
    fn dim(&self) -> usize {
        self.k
    }

    fn input_dim(&self) -> usize {
        1
    }

    fn evaluate(&self, x: ArrayView2<f64>) -> Array2<f64> {
        debug_assert_eq!(x.ncols(), 1, "CrSpline is univariate; pass shape (n, 1)");
        let n = x.nrows();
        let k = self.k;
        let num_int = k - 1;
        let co = self.build_coeffs();

        let mut design = Array2::<f64>::zeros((n, k));
        let knot_first = self.knots[0];
        let knot_last = self.knots[num_int];
        for i in 0..n {
            let xi = x[[i, 0]];
            let mut row = design.row_mut(i);
            if xi < knot_first {
                let dx = xi - knot_first;
                for j in 0..k {
                    row[j] = co.vals_0[j] + co.left_slope[j] * dx;
                }
            } else if xi > knot_last {
                let dx = xi - knot_last;
                for j in 0..k {
                    row[j] = co.vals_last[j] + co.right_slope[j] * dx;
                }
            } else {
                let lo = find_interval(&self.knots, xi, num_int);
                let dx = xi - self.knots[lo];
                let dx2 = dx * dx;
                let dx3 = dx2 * dx;
                let base = lo * k;
                for j in 0..k {
                    row[j] = co.vals_at[base + j]
                        + co.b_co[base + j] * dx
                        + co.c_co[base + j] * dx2
                        + co.d_co[base + j] * dx3;
                }
            }
        }
        design
    }

    fn d1(&self, x: ArrayView2<f64>, axis: usize) -> Array2<f64> {
        // `axis` is the input dimension. CR spline is univariate, so only
        // axis = 0 is valid (the panic guards against tensor-product callers
        // wiring the wrong dim).
        assert_eq!(axis, 0, "CrSpline is univariate; only axis=0 is valid");
        debug_assert_eq!(x.ncols(), 1, "CrSpline is univariate; pass shape (n, 1)");
        let n = x.nrows();
        let k = self.k;
        let num_int = k - 1;
        let co = self.build_coeffs();

        // Derivative of cubic `vals + b·dx + c·dx² + d·dx³` is
        //   b + 2c·dx + 3d·dx². At the boundaries the basis is extrapolated
        // linearly with slope `left_slope[j]` / `right_slope[j]`, so d1 is
        // exactly that constant outside [knot_first, knot_last].
        let mut deriv = Array2::<f64>::zeros((n, k));
        let knot_first = self.knots[0];
        let knot_last = self.knots[num_int];
        for i in 0..n {
            let xi = x[[i, 0]];
            let mut row = deriv.row_mut(i);
            if xi < knot_first {
                for j in 0..k {
                    row[j] = co.left_slope[j];
                }
            } else if xi > knot_last {
                for j in 0..k {
                    row[j] = co.right_slope[j];
                }
            } else {
                let lo = find_interval(&self.knots, xi, num_int);
                let dx = xi - self.knots[lo];
                let base = lo * k;
                for j in 0..k {
                    row[j] = co.b_co[base + j]
                        + 2.0 * co.c_co[base + j] * dx
                        + 3.0 * co.d_co[base + j] * dx * dx;
                }
            }
        }
        deriv
    }

    fn penalties(&self) -> Vec<Array2<f64>> {
        vec![cr_spline_penalty(self.knots.view())]
    }
}

/// Binary-search the half-open interval `[knots[i], knots[i+1])` that
/// contains `xi`. Caller must have already checked `xi ∈ [knots[0],
/// knots[k-1]]` — the routine returns `num_int - 1` for the right endpoint.
fn find_interval(knots: &Array1<f64>, xi: f64, num_int: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = num_int;
    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if xi < knots[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

/// `S = D' B⁻¹ D` per Wood (2006) §4.1.2. Port of v0.x
/// `src/penalty.rs::cr_spline_penalty`.
fn cr_spline_penalty(knots: ArrayView1<f64>) -> Array2<f64> {
    let n = knots.len();
    debug_assert!(n >= 3, "cr penalty needs k ≥ 3");
    let n2 = n - 2;
    let h: Vec<f64> = (0..n - 1).map(|i| knots[i + 1] - knots[i]).collect();

    let mut d_mat = Array2::<f64>::zeros((n2, n));
    for i in 0..n2 {
        d_mat[[i, i]] = 1.0 / h[i];
        d_mat[[i, i + 1]] = -1.0 / h[i] - 1.0 / h[i + 1];
        d_mat[[i, i + 2]] = 1.0 / h[i + 1];
    }

    let mut b_diag = vec![0.0; n2];
    let mut b_off = vec![0.0; n2.saturating_sub(1)];
    for i in 0..n2 {
        b_diag[i] = (h[i] + h[i + 1]) / 3.0;
    }
    for i in 0..n2.saturating_sub(1) {
        b_off[i] = h[i + 1] / 6.0;
    }

    // Solve B · X = D for X column by column with symmetric Thomas.
    let mut b_inv_d = Array2::<f64>::zeros((n2, n));
    for col in 0..n {
        let rhs: Vec<f64> = (0..n2).map(|i| d_mat[[i, col]]).collect();
        let sol = thomas_symmetric(&b_diag, &b_off, &rhs);
        for i in 0..n2 {
            b_inv_d[[i, col]] = sol[i];
        }
    }
    d_mat.t().dot(&b_inv_d)
}

/// Symmetric Thomas algorithm for a positive-definite tridiagonal system.
fn thomas_symmetric(diag: &[f64], off: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut a = diag.to_vec();
    let mut b = rhs.to_vec();
    for i in 1..n {
        let m = off[i - 1] / a[i - 1];
        a[i] -= m * off[i - 1];
        b[i] -= m * b[i - 1];
    }
    let mut x = vec![0.0; n];
    x[n - 1] = b[n - 1] / a[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = (b[i] - off[i] * x[i + 1]) / a[i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// 2-D view `(n, 1)` from a 1-D array — Phase 0 univariate helper.
    fn col(x: &Array1<f64>) -> Array2<f64> {
        x.view().insert_axis(ndarray::Axis(1)).to_owned()
    }

    #[test]
    fn evaluate_cardinal_property() {
        // At each knot, basis j = 1 and others = 0.
        let knots = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let cr = CrSpline::new(knots.clone()).unwrap();
        let x2 = col(&knots);
        let design = cr.evaluate(x2.view());
        for j in 0..5 {
            for i in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = design[[i, j]];
                assert!((got - expected).abs() < 1e-12, "i={i} j={j} got={got}");
            }
        }
    }

    #[test]
    fn rows_sum_to_one_constant_reproduced() {
        // CR spline reproduces constants: sum over j of basis_j(x) = 1.
        let knots = array![0.0, 0.3, 0.7, 1.0];
        let cr = CrSpline::new(knots).unwrap();
        let x = Array1::linspace(0.0, 1.0, 11);
        let x2 = col(&x);
        let design = cr.evaluate(x2.view());
        for i in 0..11 {
            let s: f64 = design.row(i).iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "row {i} sums to {s}");
        }
    }

    #[test]
    fn d1_matches_finite_difference() {
        // Analytical d1 must match a central-FD probe of `evaluate`.
        let knots = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let cr = CrSpline::new(knots).unwrap();
        // Sample interior points, then a couple just outside the boundaries.
        let xs = array![0.05, 0.13, 0.4, 0.62, 0.87, -0.1, 1.2];
        let x2 = col(&xs);
        let d1 = cr.d1(x2.view(), 0);

        let h = 1e-5;
        for i in 0..xs.len() {
            let xi = xs[i];
            let x_plus = col(&array![xi + h]);
            let x_minus = col(&array![xi - h]);
            let f_plus = cr.evaluate(x_plus.view());
            let f_minus = cr.evaluate(x_minus.view());
            for j in 0..5 {
                let fd = (f_plus[[0, j]] - f_minus[[0, j]]) / (2.0 * h);
                assert!(
                    (d1[[i, j]] - fd).abs() < 1e-7,
                    "row {i} (x={xi}) col {j}: d1={} fd={fd}",
                    d1[[i, j]],
                );
            }
        }
    }

    #[test]
    fn d1_of_constant_basis_sum_is_zero() {
        // ∑_j basis_j(x) = 1 everywhere, so ∂/∂x ∑_j = 0.
        let knots = array![0.0, 0.3, 0.7, 1.0];
        let cr = CrSpline::new(knots).unwrap();
        let x = Array1::linspace(0.0, 1.0, 9);
        let x2 = col(&x);
        let d1 = cr.d1(x2.view(), 0);
        for i in 0..9 {
            let s: f64 = d1.row(i).iter().sum();
            assert!(s.abs() < 1e-10, "row {i} d1 sums to {s}");
        }
    }

    #[test]
    fn penalty_is_symmetric_positive_semidefinite() {
        let knots = array![0.0, 0.2, 0.5, 0.8, 1.0];
        let s = cr_spline_penalty(knots.view());
        // Symmetry
        for i in 0..5 {
            for j in 0..5 {
                assert!((s[[i, j]] - s[[j, i]]).abs() < 1e-12);
            }
        }
        // Constant vector is in nullspace (penalty on second derivatives).
        let ones = Array1::ones(5);
        let s_ones = s.dot(&ones);
        for v in s_ones.iter() {
            assert!(v.abs() < 1e-10, "S·1 should be 0, got {v}");
        }
        // Linear vector also in nullspace.
        let lin = knots.clone();
        let s_lin = s.dot(&lin);
        for v in s_lin.iter() {
            assert!(v.abs() < 1e-9, "S·linear should be 0, got {v}");
        }
    }
}
