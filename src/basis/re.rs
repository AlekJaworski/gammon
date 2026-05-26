//! Random-effect basis (`bs="re"` in mgcv).
//!
//! Models a categorical predictor as independent random effects: the design
//! matrix is a one-hot indicator matrix (one column per unique level), and
//! the penalty is the identity matrix I_p. The smoothing parameter λ acts
//! as the inverse variance of the Gaussian prior on each level's coefficient.
//!
//! Levels are stored as `Vec<f64>` — cluster IDs in this codebase are
//! integer-valued floats (1.0, 2.0, …). Exact bit equality is used for the
//! lookup because the values are integer-valued.
//!
//! Prediction on unseen levels returns a row of zeros (the smooth
//! contributes nothing for unseen groups — matches mgcv's behaviour).
//!
//! Verbatim port of v0.x `src/basis.rs::RandomEffectBasis` (lines 945-1025)
//! into gammon's `Basis` trait shape.

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::traits::Basis;

/// Random-effect basis. The number of basis functions equals the number of
/// unique training levels; the penalty is the identity.
pub struct RandomEffectsBasis {
    /// Unique sorted levels from training data. Dedup uses exact float-bit
    /// equality (appropriate for integer-valued category IDs).
    pub levels: Vec<f64>,
}

impl RandomEffectsBasis {
    /// Build a `RandomEffectsBasis` from raw training-data grouping values.
    ///
    /// Collects unique values, sorts them, and stores them as the level
    /// vocabulary. Duplicate detection uses exact float-bit equality
    /// (appropriate for integer-valued category IDs).
    pub fn from_data(x: ArrayView1<f64>) -> Self {
        let mut vals: Vec<f64> = x.iter().copied().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup_by(|a, b| a.to_bits() == b.to_bits());
        Self { levels: vals }
    }
}

impl Basis for RandomEffectsBasis {
    fn dim(&self) -> usize {
        self.levels.len()
    }

    fn input_dim(&self) -> usize {
        1
    }

    /// One-hot design: `Z[i, j] = 1` iff `x[i, 0] == levels[j]`, else 0.
    /// Unseen levels → row of zeros (matches mgcv's behaviour on new data).
    fn evaluate(&self, x: ArrayView2<f64>) -> Array2<f64> {
        debug_assert_eq!(
            x.ncols(),
            1,
            "RandomEffectsBasis is univariate; pass shape (n, 1)"
        );
        let n = x.nrows();
        let p = self.levels.len();
        let mut z = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let xi = x[[i, 0]];
            // Binary search on sorted levels for an exact bit match.
            let pos = self.levels.partition_point(|&lv| {
                lv.partial_cmp(&xi).unwrap_or(std::cmp::Ordering::Less) == std::cmp::Ordering::Less
            });
            if pos < p && self.levels[pos].to_bits() == xi.to_bits() {
                z[[i, pos]] = 1.0;
            }
            // else: unseen level → row stays zero.
        }
        z
    }

    /// Random effects have no smooth-coordinate derivative — the design is
    /// a step function over the grouping variable. Return a zero matrix of
    /// the right shape so tensor-product / chain-rule callers can compose
    /// without special-casing. Matches v0.x's predict-time derivative for
    /// `bs="re"`, which is identically zero.
    fn d1(&self, x: ArrayView2<f64>, _axis: usize) -> Array2<f64> {
        debug_assert_eq!(
            x.ncols(),
            1,
            "RandomEffectsBasis is univariate; pass shape (n, 1)"
        );
        Array2::<f64>::zeros((x.nrows(), self.levels.len()))
    }

    fn penalties(&self) -> Vec<Array2<f64>> {
        vec![Array2::<f64>::eye(self.dim())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    /// 2-D view `(n, 1)` from a 1-D array — univariate helper.
    fn col(x: &Array1<f64>) -> Array2<f64> {
        x.view().insert_axis(ndarray::Axis(1)).to_owned()
    }

    #[test]
    fn re_basis_from_data_sorts_and_dedups() {
        let x = array![1.0, 3.0, 1.0, 2.0];
        let re = RandomEffectsBasis::from_data(x.view());
        assert_eq!(re.levels, vec![1.0, 2.0, 3.0]);
        assert_eq!(re.dim(), 3);
        assert_eq!(re.input_dim(), 1);
    }

    #[test]
    fn re_evaluate_is_one_hot() {
        let train = array![1.0, 2.0, 3.0];
        let re = RandomEffectsBasis::from_data(train.view());
        // Evaluate at known levels in non-sorted order.
        let x = array![3.0, 1.0, 2.0, 2.0];
        let z = re.evaluate(col(&x).view());
        assert_eq!(z.shape(), &[4, 3]);
        // Row 0: x=3.0 → column index 2.
        // Row 1: x=1.0 → column index 0.
        // Row 2: x=2.0 → column index 1.
        // Row 3: x=2.0 → column index 1.
        let expected = array![
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(
                    z[[i, j]],
                    expected[[i, j]],
                    "row {i} col {j}: expected {} got {}",
                    expected[[i, j]],
                    z[[i, j]]
                );
            }
            // Exactly one 1 per row.
            let s: f64 = z.row(i).iter().sum();
            assert_eq!(s, 1.0, "row {i} should have exactly one 1");
        }
    }

    #[test]
    fn re_unseen_level_yields_zero_row() {
        let train = array![1.0, 2.0, 3.0];
        let re = RandomEffectsBasis::from_data(train.view());
        // Predict on x containing level 5.0 (not in training) plus a known.
        let x = array![5.0, 2.0];
        let z = re.evaluate(col(&x).view());
        // Row 0 (unseen 5.0): all zeros.
        for j in 0..3 {
            assert_eq!(z[[0, j]], 0.0, "unseen row col {j} should be 0");
        }
        // Row 1 (seen 2.0): one-hot at column 1.
        assert_eq!(z[[1, 0]], 0.0);
        assert_eq!(z[[1, 1]], 1.0);
        assert_eq!(z[[1, 2]], 0.0);
    }

    #[test]
    fn re_penalty_is_identity() {
        let train = array![1.0, 2.0, 3.0, 4.0];
        let re = RandomEffectsBasis::from_data(train.view());
        let pens = re.penalties();
        assert_eq!(pens.len(), 1, "RE basis has exactly one penalty");
        let s = &pens[0];
        assert_eq!(s.shape(), &[4, 4]);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(s[[i, j]], expected, "S[{i},{j}] should be {expected}");
            }
        }
    }

    #[test]
    fn re_d1_returns_zeros() {
        let train = array![1.0, 2.0, 3.0];
        let re = RandomEffectsBasis::from_data(train.view());
        let x = array![1.0, 2.0, 3.0, 5.0, 2.5];
        let d = re.d1(col(&x).view(), 0);
        assert_eq!(d.shape(), &[5, 3]);
        for i in 0..5 {
            for j in 0..3 {
                assert_eq!(d[[i, j]], 0.0, "d1[{i},{j}] should be 0");
            }
        }
    }
}
