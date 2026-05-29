"""Generate the quantile OOS-pinball parity fixture against mgcv_rust.

gamrs's quantile (ELF / qgam-style) fits target the same OOS pinball quality
as the mature `mgcv_rust` engine. This script fits mgcv_rust's fast-OOS
quantile path on a fixed heteroscedastic split and records, per τ, its
out-of-sample pinball loss + test predictions. The gamrs parity test
(tests/python/test_parity_quantile.py) then fits gamrs on the identical split
and asserts its OOS pinball is on par.

Requires `mgcv_rust` installed in the venv (PyPI: `pip install mgcv_rust`).
It is NOT a gamrs runtime/test dependency — only this generator needs it, so
the committed fixture keeps the parity test self-contained.

    python scripts/gen_quantile_oos_reference.py
"""
import json
from pathlib import Path

import numpy as np

import mgcv_rust as mr

SEED = 20260529
N_TR, N_TE = 800, 400
TAUS = [0.1, 0.3, 0.5, 0.7, 0.9]
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "quantile_oos_hetero_n800_cr.json"


def _gen_y(x, rng):
    # Heteroscedastic: noise scale grows with x, so the quantile spread is
    # x-dependent — a non-trivial target for a smooth quantile fit.
    return np.sin(2 * np.pi * x) + (0.2 + 0.6 * x) * rng.standard_normal(x.size)


def _pinball(y, q, tau):
    r = y - q
    return float(np.maximum(tau * r, (tau - 1.0) * r).mean())


def main():
    rng = np.random.default_rng(SEED)
    x_tr = rng.uniform(0, 1, N_TR)
    x_te = np.linspace(0, 1, N_TE)
    y_tr = _gen_y(x_tr, rng)
    y_te = _gen_y(x_te, rng)

    per_tau = {}
    for tau in TAUS:
        gam, sigma, _info = mr.fit_quantile(
            x_tr.reshape(-1, 1), y_tr, tau, k=[10], preset="fast_oos"
        )
        q_te = np.asarray(gam.predict(x_te.reshape(-1, 1))).ravel()
        pb = _pinball(y_te, q_te, tau)
        per_tau[str(tau)] = {"sigma": float(sigma), "oos_pinball": pb, "pred_test": q_te.tolist()}
        print(f"  tau={tau}: mgcv_rust OOS pinball={pb:.6f} sigma={sigma:.4f}")

    fixture = {
        "schema_version": 1,
        "name": "quantile_oos_hetero_n800_cr",
        "description": "Heteroscedastic 1-D quantile OOS-pinball reference (mgcv_rust fast_oos)",
        "metadata": {"engine": "mgcv_rust", "engine_version": getattr(mr, "__version__", "0.23.2")},
        "inputs": {
            "seed": SEED, "n_train": N_TR, "n_test": N_TE, "k": [10], "bs": ["cr"], "taus": TAUS,
            "x_train": x_tr.tolist(), "y_train": y_tr.tolist(),
            "x_test": x_te.tolist(), "y_test": y_te.tolist(),
        },
        "mgcv_rust": {"per_tau": per_tau},
    }
    OUT.write_text(json.dumps(fixture))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
