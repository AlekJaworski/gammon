"""`GamPredictor` — frozen, inference-only view of a fitted gammon :class:`Gam`.

Mirrors :class:`mgcv_rust.GamPredictor` in shape and intent: wraps a
fitted :class:`Gam` via composition and exposes only the inference API
(``predict`` / ``predict_ci`` / ``predict_diff`` / sklearn-style
attrs). Construct one from a fitted Gam — or restore one straight from
disk via :meth:`GamPredictor.load`.

Use this in deployment / serving paths where you want:

- **A frozen API surface** — no ``fit``, no constructor knobs to
  twiddle; ``__slots__`` blocks accidental mutation.
- **Round-trip verification** — :meth:`check_against` asserts the
  predictor reproduces a reference :class:`Gam`'s output exactly on a
  sample, catching serialization or mismatch bugs.
- **Save/load** — :meth:`save` and :meth:`load` route through
  :meth:`Gam.save` / :meth:`Gam.load` so the same on-disk format works
  for both wrappers.
"""

from __future__ import annotations

import os
from typing import Any, Union

import numpy as np

from ._fitter import ArrayLike, Gam, TermContributions


class GamPredictor:
    """Frozen, inference-only view of a fitted :class:`Gam`.

    Wraps a fitted Gam via composition; exposes only the inference-time
    API. Once built, the bound Gam's coefficients, vcov, and feature
    schema are the contract — mutating the predictor is disallowed
    (``__slots__`` + no ``fit``).
    """

    __slots__ = ("_gam",)

    def __init__(self, gam: Gam) -> None:
        if not isinstance(gam, Gam):
            raise TypeError(
                f"GamPredictor expects a Gam, got {type(gam).__name__}"
            )
        if gam._fitted is None:
            raise RuntimeError(
                "GamPredictor requires a fitted Gam — call .fit() first."
            )
        # Use object.__setattr__ to bypass __slots__ semantics during construction.
        object.__setattr__(self, "_gam", gam)

    # ------------------------- sklearn-style attrs ------------------------- #

    @property
    def feature_names_in_(self) -> np.ndarray:
        return self._gam.feature_names_in_

    @property
    def n_features_in_(self) -> int:
        return self._gam.n_features_in_

    @property
    def coef_(self) -> np.ndarray:
        return self._gam.coef_

    @property
    def intercept_(self) -> float:
        return self._gam.intercept_

    @property
    def intercept_response_(self) -> float:
        return self._gam.intercept_response_

    @property
    def lambda_(self) -> np.ndarray:
        return self._gam.lambda_

    @property
    def vcov_(self) -> np.ndarray:
        return self._gam.vcov_

    @property
    def k_(self) -> np.ndarray:
        return self._gam.k_

    @property
    def bs_(self) -> np.ndarray:
        return self._gam.bs_

    @property
    def edf_(self) -> np.ndarray:
        return self._gam.edf_

    @property
    def family(self) -> str:
        return self._gam.family

    @property
    def link(self) -> str:
        return self._gam.link

    # ------------------------------ Predict ------------------------------- #

    def predict(
        self,
        X: ArrayLike,
        scale: str = "response",
        type: Any = None,
    ) -> Union[np.ndarray, TermContributions]:
        """Delegates to :meth:`Gam.predict`."""
        return self._gam.predict(X, scale=scale, type=type)

    def predict_ci(
        self,
        X: ArrayLike,
        alpha: Any = None,
        n_samples: int = 1000,
        predictor: Any = None,
        seed: int = 42,
        *,
        level: float = 0.95,
        scale: str = "response",
    ) -> tuple:
        """Delegates to :meth:`Gam.predict_ci`."""
        return self._gam.predict_ci(
            X,
            alpha=alpha,
            n_samples=n_samples,
            predictor=predictor,
            seed=seed,
            level=level,
            scale=scale,
        )

    def predict_diff(
        self,
        from_X: ArrayLike,
        to_X: ArrayLike,
        level: Any = None,
        broadcast: str = "none",
        n_samples: int = 1000,
        seed: int = 42,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Delegates to :meth:`Gam.predict_diff`."""
        return self._gam.predict_diff(
            from_X,
            to_X,
            level=level,
            broadcast=broadcast,
            n_samples=n_samples,
            seed=seed,
        )

    # ---------------------------- Persistence ----------------------------- #

    def save(self, path: Union[str, "os.PathLike[str]"]) -> None:
        """Persist the underlying :class:`Gam` to disk. Pairs with
        :meth:`GamPredictor.load`."""
        self._gam.save(path)

    @classmethod
    def load(cls, path: Union[str, "os.PathLike[str]"]) -> "GamPredictor":
        """Construct a :class:`GamPredictor` from a file written by
        :meth:`Gam.save` (or :meth:`GamPredictor.save`)."""
        gam = Gam.load(path)
        return cls(gam)

    # --------------------------- Round-trip check ------------------------- #

    def check_against(
        self,
        gam: Gam,
        X_sample: ArrayLike,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        """Assert this predictor matches ``gam.predict`` on ``X_sample``.

        Use at deployment time to catch:
        - The predictor was built from a different fit than expected.
        - The bound :class:`Gam`'s state has drifted between
          serialization and load.

        Args:
            gam: a :class:`Gam` whose output should match this predictor's.
            X_sample: a small batch of input rows.
            rtol / atol: passed to :func:`numpy.allclose`.

        Raises:
            AssertionError: if any prediction diverges beyond
                ``(rtol, atol)``. The error message includes the max
                absolute and relative gap so the call site can decide
                whether to fail-closed or warn.
        """
        ours = np.asarray(self.predict(X_sample), dtype=float)
        theirs = np.asarray(gam.predict(X_sample), dtype=float)
        if not np.allclose(ours, theirs, rtol=rtol, atol=atol):
            abs_err = float(np.max(np.abs(ours - theirs)))
            denom = np.where(theirs == 0, 1.0, theirs)
            rel_err = float(np.max(np.abs((ours - theirs) / denom)))
            raise AssertionError(
                f"GamPredictor predictions diverge from Gam: max abs err "
                f"{abs_err:.3e}, max rel err {rel_err:.3e} "
                f"(rtol={rtol}, atol={atol}). The predictor may have been "
                "built from a different fit, or the bound Gam's state has "
                "drifted."
            )

    # ------------------------------- Repr --------------------------------- #

    def __repr__(self) -> str:
        names = list(self.feature_names_in_) if self.feature_names_in_.size else []
        return (
            f"GamPredictor(family={self.family!r}, link={self.link!r}, "
            f"features={names})"
        )
