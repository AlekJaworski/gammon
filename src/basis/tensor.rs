//! Tensor-product basis (mgcv `te(x0, x1)`).
//!
//! Anisotropic 2-margin tensor product. For univariate marginal bases
//! `A` and `B` with column counts `k_a` and `k_b`, the tensor product
//! basis has `k_a · k_b` columns. On a point `(x_a, x_b)`:
//!
//! ```text
//!   X_te[i, j_a * k_b + j_b] = X_a[i, j_a] * X_b[i, j_b]
//! ```
//!
//! i.e. row `i` is `kron(X_a[i, :], X_b[i, :])` (mgcv's
//! `tensor.prod.model.matrix`, src/mat.c `mgcv_tensor_mm`). The column
//! layout follows mgcv exactly: margin A indexes the "outer" loop, margin
//! B the "inner".
//!
//! Per-margin penalties (mgcv `tensor.prod.penalties`):
//!
//! ```text
//!   S_te[0] = S_a ⊗ I_{k_b}   (penalises wiggliness along margin A)
//!   S_te[1] = I_{k_a} ⊗ S_b   (penalises wiggliness along margin B)
//! ```
//!
//! Two smoothing parameters per `te(...)` term — the multi-d outer Newton
//! from 94b handles them via the per-term `Vec<Array2>` penalty list.
//!
//! Marginal centring (sum-to-zero) is applied to each margin BEFORE the
//! Kronecker product — composing `SumToZero<CrSpline>` for each margin is
//! the canonical path. The tensor product itself is NOT centred (centring
//! the product would induce a rank deficiency without resolving the
//! intercept-collinearity the way mgcv does).
//!
//! Wood (2017) §5.6 gives the math. The current scope (epic 94c) is
//! 2-margin only; n-margin generalisation is a future variant.

use ndarray::{Array2, ArrayView2};

use crate::traits::Basis;

/// Anisotropic 2-margin tensor product of marginal bases `A` and `B`.
///
/// `col_a` / `col_b` identify which columns of the input `x` each margin
/// reads. Each margin sees a single-column slice `(n, 1)`, mirroring the
/// existing univariate basis evaluators. The Kronecker is taken row-wise
/// (matches mgcv's `tensor.prod.model.matrix` column layout exactly:
/// outer loop margin A, inner loop margin B).
///
/// Predictions are basis-invariant w.r.t. margin permutation (swapping
/// `(A, B)` for `(B, A)` permutes columns of the design and the per-
/// margin penalties symmetrically) — but the natural canonical ordering
/// `(margin A first, margin B inside)` matches mgcv, so callers should
/// pass `(margin_a, margin_b)` in the same order they appear in the
/// equivalent `te(x_a, x_b)` formula.
pub struct TensorProductBasis<A, B> {
    pub margin_a: A,
    pub margin_b: B,
    pub col_a: usize,
    pub col_b: usize,
}

impl<A: Basis, B: Basis> TensorProductBasis<A, B> {
    /// Construct from already-built marginal bases. The marginals are
    /// typically `SumToZero<CrSpline>` (each centred independently) —
    /// see [`crate::design::additive::TermSpec::Tensor`] for the
    /// canonical fit-time wiring.
    pub fn new(margin_a: A, margin_b: B, col_a: usize, col_b: usize) -> Self {
        Self {
            margin_a,
            margin_b,
            col_a,
            col_b,
        }
    }

    /// Number of columns from margin A.
    pub fn dim_a(&self) -> usize {
        self.margin_a.dim()
    }

    /// Number of columns from margin B.
    pub fn dim_b(&self) -> usize {
        self.margin_b.dim()
    }
}

impl<A: Basis, B: Basis> Basis for TensorProductBasis<A, B> {
    fn dim(&self) -> usize {
        self.margin_a.dim() * self.margin_b.dim()
    }

    fn input_dim(&self) -> usize {
        2
    }

    /// Row-wise Kronecker product of the two marginal designs.
    /// `x` has shape `(n, n_input_dims)`; margin A reads `x.column(col_a)`,
    /// margin B reads `x.column(col_b)`. Result has shape `(n, k_a*k_b)`
    /// with `result[i, j_a*k_b + j_b] = X_a[i, j_a] * X_b[i, j_b]`.
    fn evaluate(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let x_a = x.slice(ndarray::s![.., self.col_a..self.col_a + 1]);
        let x_b = x.slice(ndarray::s![.., self.col_b..self.col_b + 1]);
        let design_a = self.margin_a.evaluate(x_a);
        let design_b = self.margin_b.evaluate(x_b);
        let k_a = design_a.ncols();
        let k_b = design_b.ncols();
        debug_assert_eq!(design_a.nrows(), n);
        debug_assert_eq!(design_b.nrows(), n);
        let mut out = Array2::<f64>::zeros((n, k_a * k_b));
        for i in 0..n {
            for j_a in 0..k_a {
                let a_val = design_a[[i, j_a]];
                let base = j_a * k_b;
                for j_b in 0..k_b {
                    out[[i, base + j_b]] = a_val * design_b[[i, j_b]];
                }
            }
        }
        out
    }

    /// `∂design/∂x_axis` via the product rule on the row-wise Kronecker.
    /// Only `axis == col_a` and `axis == col_b` give non-zero derivatives;
    /// any other axis is treated as constant in this basis (returns the
    /// zero matrix).
    ///
    /// For `axis == col_a`: `∂(X_a · X_b)/∂x_a = (∂X_a/∂x_a) ⊗ X_b`.
    /// For `axis == col_b`: `∂(X_a · X_b)/∂x_b = X_a ⊗ (∂X_b/∂x_b)`.
    fn d1(&self, x: ArrayView2<f64>, axis: usize) -> Array2<f64> {
        let n = x.nrows();
        let k_a = self.margin_a.dim();
        let k_b = self.margin_b.dim();
        let mut out = Array2::<f64>::zeros((n, k_a * k_b));

        if axis == self.col_a {
            let x_a = x.slice(ndarray::s![.., self.col_a..self.col_a + 1]);
            let x_b = x.slice(ndarray::s![.., self.col_b..self.col_b + 1]);
            // Univariate marginal bases ignore their `axis` parameter
            // (they take a single column slice) so passing `axis=0` is
            // correct here.
            let d_a = self.margin_a.d1(x_a, 0);
            let design_b = self.margin_b.evaluate(x_b);
            for i in 0..n {
                for j_a in 0..k_a {
                    let a_val = d_a[[i, j_a]];
                    let base = j_a * k_b;
                    for j_b in 0..k_b {
                        out[[i, base + j_b]] = a_val * design_b[[i, j_b]];
                    }
                }
            }
        } else if axis == self.col_b {
            let x_a = x.slice(ndarray::s![.., self.col_a..self.col_a + 1]);
            let x_b = x.slice(ndarray::s![.., self.col_b..self.col_b + 1]);
            let design_a = self.margin_a.evaluate(x_a);
            let d_b = self.margin_b.d1(x_b, 0);
            for i in 0..n {
                for j_a in 0..k_a {
                    let a_val = design_a[[i, j_a]];
                    let base = j_a * k_b;
                    for j_b in 0..k_b {
                        out[[i, base + j_b]] = a_val * d_b[[i, j_b]];
                    }
                }
            }
        }
        out
    }

    /// Per-margin penalties lifted to the `(k_a*k_b, k_a*k_b)` space:
    /// `[S_a ⊗ I_{k_b}, I_{k_a} ⊗ S_b]`. Two penalties → two smoothing
    /// parameters per `te(...)` term.
    ///
    /// Uses each margin's own `penalties()` (so a `SumToZero<CrSpline>`
    /// margin contributes its centred penalty, which is the matrix the
    /// inner solver needs).
    fn penalties(&self) -> Vec<Array2<f64>> {
        let s_a_list = self.margin_a.penalties();
        let s_b_list = self.margin_b.penalties();
        debug_assert_eq!(
            s_a_list.len(),
            1,
            "tensor product currently only supports marginals with one penalty"
        );
        debug_assert_eq!(
            s_b_list.len(),
            1,
            "tensor product currently only supports marginals with one penalty"
        );
        let s_a = &s_a_list[0];
        let s_b = &s_b_list[0];
        let k_a = self.margin_a.dim();
        let k_b = self.margin_b.dim();
        debug_assert_eq!(s_a.nrows(), k_a);
        debug_assert_eq!(s_a.ncols(), k_a);
        debug_assert_eq!(s_b.nrows(), k_b);
        debug_assert_eq!(s_b.ncols(), k_b);

        // S_te_a[(j_a*k_b + k1), (l_a*k_b + k2)] = S_a[j_a, l_a] · δ(k1, k2)
        let mut s_te_a = Array2::<f64>::zeros((k_a * k_b, k_a * k_b));
        for j_a in 0..k_a {
            for l_a in 0..k_a {
                let s_val = s_a[[j_a, l_a]];
                if s_val == 0.0 {
                    continue;
                }
                let row_base = j_a * k_b;
                let col_base = l_a * k_b;
                for k in 0..k_b {
                    s_te_a[[row_base + k, col_base + k]] = s_val;
                }
            }
        }

        // S_te_b[(j_a*k_b + k1), (l_a*k_b + k2)] = δ(j_a, l_a) · S_b[k1, k2]
        let mut s_te_b = Array2::<f64>::zeros((k_a * k_b, k_a * k_b));
        for j_a in 0..k_a {
            let base = j_a * k_b;
            for k1 in 0..k_b {
                for k2 in 0..k_b {
                    s_te_b[[base + k1, base + k2]] = s_b[[k1, k2]];
                }
            }
        }

        vec![s_te_a, s_te_b]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::CrSpline;
    use crate::transform::SumToZero;
    use ndarray::{array, Array1, Array2};

    /// Build a `(n, 2)` input from two 1-D vectors.
    fn stack_xy(x: &Array1<f64>, y: &Array1<f64>) -> Array2<f64> {
        let n = x.len();
        debug_assert_eq!(y.len(), n);
        let mut out = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            out[[i, 0]] = x[i];
            out[[i, 1]] = y[i];
        }
        out
    }

    /// Helper: build a centred CR spline marginal from a 1-D x vector.
    fn build_centred_cr(k: usize, x: &Array1<f64>) -> SumToZero<CrSpline> {
        let cr = CrSpline::with_quantile_knots(x.view(), k).unwrap();
        let x_2d = x.view().insert_axis(ndarray::Axis(1)).to_owned();
        let raw = cr.evaluate(x_2d.view());
        SumToZero::from_fit_design(cr, raw.view())
    }

    /// Shape check on the tensor product of two centred CR margins.
    #[test]
    fn tensor_design_shape_and_penalties() {
        let n = 50;
        let x_a: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let x_b: Array1<f64> = (0..n).map(|i| (i as f64 * 0.13).sin()).collect();

        let marg_a = build_centred_cr(5, &x_a);
        let marg_b = build_centred_cr(4, &x_b);
        let k_a = marg_a.dim(); // 4 after centring
        let k_b = marg_b.dim(); // 3 after centring

        let xy = stack_xy(&x_a, &x_b);
        let te = TensorProductBasis::new(marg_a, marg_b, 0, 1);
        assert_eq!(te.dim(), k_a * k_b);
        assert_eq!(te.input_dim(), 2);

        let design = te.evaluate(xy.view());
        assert_eq!(design.shape(), &[n, k_a * k_b]);

        let pens = te.penalties();
        assert_eq!(pens.len(), 2, "tensor product has 2 penalties");
        assert_eq!(pens[0].shape(), &[k_a * k_b, k_a * k_b]);
        assert_eq!(pens[1].shape(), &[k_a * k_b, k_a * k_b]);
    }

    /// Row-wise Kronecker layout: column `j_a*k_b + j_b` should equal
    /// `X_a[:, j_a] * X_b[:, j_b]` element-wise.
    #[test]
    fn tensor_design_matches_row_kronecker() {
        let n = 30;
        let x_a: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let x_b: Array1<f64> = Array1::linspace(0.0, 1.0, n).mapv(|v: f64| (v * 2.0 + 0.1).ln());

        let marg_a = CrSpline::with_quantile_knots(x_a.view(), 5).unwrap();
        let marg_b = CrSpline::with_quantile_knots(x_b.view(), 4).unwrap();
        let k_a = marg_a.dim();
        let k_b = marg_b.dim();

        let xy = stack_xy(&x_a, &x_b);
        let xa_2d = x_a.view().insert_axis(ndarray::Axis(1)).to_owned();
        let xb_2d = x_b.view().insert_axis(ndarray::Axis(1)).to_owned();
        let design_a = marg_a.evaluate(xa_2d.view());
        let design_b = marg_b.evaluate(xb_2d.view());

        let te = TensorProductBasis::new(marg_a, marg_b, 0, 1);
        let design = te.evaluate(xy.view());

        for i in 0..n {
            for j_a in 0..k_a {
                for j_b in 0..k_b {
                    let expected = design_a[[i, j_a]] * design_b[[i, j_b]];
                    let got = design[[i, j_a * k_b + j_b]];
                    assert!(
                        (got - expected).abs() < 1e-12,
                        "row {i} (j_a={j_a}, j_b={j_b}): got {got} expected {expected}",
                    );
                }
            }
        }
    }

    /// Penalty `S_te_0 = S_a ⊗ I_{k_b}` should be block-diagonal-ish:
    /// nonzero entries only when the "inner" index matches.
    /// Penalty `S_te_1 = I_{k_a} ⊗ S_b` should be block-diagonal with
    /// `S_b` repeated `k_a` times along the diagonal.
    #[test]
    fn tensor_penalties_kronecker_structure() {
        let knots_a = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let knots_b = array![0.0, 0.3, 0.7, 1.0];
        let marg_a = CrSpline::new(knots_a).unwrap();
        let marg_b = CrSpline::new(knots_b).unwrap();
        let k_a = marg_a.dim();
        let k_b = marg_b.dim();
        let s_a = marg_a.penalties().pop().unwrap();
        let s_b = marg_b.penalties().pop().unwrap();

        let te = TensorProductBasis::new(marg_a, marg_b, 0, 1);
        let pens = te.penalties();
        let s_te_a = &pens[0];
        let s_te_b = &pens[1];

        // S_te_a structure: S_a ⊗ I_{k_b}
        for j_a in 0..k_a {
            for l_a in 0..k_a {
                for k1 in 0..k_b {
                    for k2 in 0..k_b {
                        let row = j_a * k_b + k1;
                        let col = l_a * k_b + k2;
                        let expected = if k1 == k2 { s_a[[j_a, l_a]] } else { 0.0 };
                        let got = s_te_a[[row, col]];
                        assert!(
                            (got - expected).abs() < 1e-14,
                            "S_te_a[{row},{col}] (j_a={j_a},l_a={l_a},k1={k1},k2={k2}): \
                             got {got} expected {expected}"
                        );
                    }
                }
            }
        }

        // S_te_b structure: I_{k_a} ⊗ S_b — block-diagonal, S_b repeated.
        for j_a in 0..k_a {
            for l_a in 0..k_a {
                for k1 in 0..k_b {
                    for k2 in 0..k_b {
                        let row = j_a * k_b + k1;
                        let col = l_a * k_b + k2;
                        let expected = if j_a == l_a { s_b[[k1, k2]] } else { 0.0 };
                        let got = s_te_b[[row, col]];
                        assert!(
                            (got - expected).abs() < 1e-14,
                            "S_te_b[{row},{col}] (j_a={j_a},l_a={l_a},k1={k1},k2={k2}): \
                             got {got} expected {expected}"
                        );
                    }
                }
            }
        }
    }

    /// `∂design/∂x_a` should equal the row-wise Kronecker of `d_a` with
    /// `design_b`. Verify with a central finite-difference probe on a few
    /// interior points.
    #[test]
    fn tensor_d1_matches_finite_difference_axis_a() {
        let n = 20;
        let x_a: Array1<f64> = Array1::linspace(0.05, 0.95, n);
        let x_b: Array1<f64> = (0..n)
            .map(|i| 0.1 + 0.8 * (i as f64) / (n as f64 - 1.0))
            .collect();
        let marg_a = CrSpline::with_quantile_knots(x_a.view(), 5).unwrap();
        let marg_b = CrSpline::with_quantile_knots(x_b.view(), 4).unwrap();
        let te = TensorProductBasis::new(marg_a, marg_b, 0, 1);

        let xy = stack_xy(&x_a, &x_b);
        let d1 = te.d1(xy.view(), 0);

        // Central FD probe at each row.
        let h = 1e-5;
        for i in 0..n {
            let mut xy_plus = xy.clone();
            let mut xy_minus = xy.clone();
            xy_plus[[i, 0]] += h;
            xy_minus[[i, 0]] -= h;
            // We only need row i — evaluating the full design is fine.
            let f_plus = te.evaluate(xy_plus.view());
            let f_minus = te.evaluate(xy_minus.view());
            for j in 0..te.dim() {
                let fd = (f_plus[[i, j]] - f_minus[[i, j]]) / (2.0 * h);
                let got = d1[[i, j]];
                assert!(
                    (got - fd).abs() < 1e-6,
                    "d1[{i},{j}] axis=0: got {got} fd={fd}"
                );
            }
        }
    }

    /// Same as above for axis = col_b.
    #[test]
    fn tensor_d1_matches_finite_difference_axis_b() {
        let n = 20;
        let x_a: Array1<f64> = Array1::linspace(0.05, 0.95, n);
        let x_b: Array1<f64> = (0..n)
            .map(|i| 0.1 + 0.8 * (i as f64) / (n as f64 - 1.0))
            .collect();
        let marg_a = CrSpline::with_quantile_knots(x_a.view(), 5).unwrap();
        let marg_b = CrSpline::with_quantile_knots(x_b.view(), 4).unwrap();
        let te = TensorProductBasis::new(marg_a, marg_b, 0, 1);

        let xy = stack_xy(&x_a, &x_b);
        let d1 = te.d1(xy.view(), 1);

        let h = 1e-5;
        for i in 0..n {
            let mut xy_plus = xy.clone();
            let mut xy_minus = xy.clone();
            xy_plus[[i, 1]] += h;
            xy_minus[[i, 1]] -= h;
            let f_plus = te.evaluate(xy_plus.view());
            let f_minus = te.evaluate(xy_minus.view());
            for j in 0..te.dim() {
                let fd = (f_plus[[i, j]] - f_minus[[i, j]]) / (2.0 * h);
                let got = d1[[i, j]];
                assert!(
                    (got - fd).abs() < 1e-6,
                    "d1[{i},{j}] axis=1: got {got} fd={fd}"
                );
            }
        }
    }

    /// Derivative w.r.t. an axis not used by the basis should be zero.
    #[test]
    fn tensor_d1_unrelated_axis_is_zero() {
        let n = 10;
        let x_a: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let x_b: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let marg_a = CrSpline::with_quantile_knots(x_a.view(), 5).unwrap();
        let marg_b = CrSpline::with_quantile_knots(x_b.view(), 4).unwrap();
        let te = TensorProductBasis::new(marg_a, marg_b, 0, 1);
        // Make a 3-column input so axis=2 is valid but unused.
        let mut xy = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            xy[[i, 0]] = x_a[i];
            xy[[i, 1]] = x_b[i];
            xy[[i, 2]] = 0.123;
        }
        let d1 = te.d1(xy.view(), 2);
        for i in 0..n {
            for j in 0..te.dim() {
                assert_eq!(d1[[i, j]], 0.0, "d1[{i},{j}] axis=2 should be 0");
            }
        }
    }
}
