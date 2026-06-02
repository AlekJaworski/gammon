#!/usr/bin/env python
"""Discretizable-data bench: gamrs vs mgcv_rust.bam (discrete=True).

bam()'s `discrete=True` mode bins predictors into a coarse grid then
scatter-gathers the design — saves O(n·p²) → O(unique·p²) on `X'WX`.
This pays off when there are MANY duplicate predictor values. Pure
continuous predictors don't benefit (and have setup overhead).

Test: integer-binned predictor with few unique values, varying n.

Usage:
    .venv/bin/python scripts/bench_discretizable.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import gamrs

try:
    import mgcv_rust as mr
except ImportError:
    mr = None


N_VALUES = [10_000, 100_000, 1_000_000]
N_UNIQUE_VALUES = [50, 200, 1000]
RUNS = 3
RNG = np.random.default_rng(42)


def synth_binned_gaussian(n: int, n_unique: int):
    """Integer-binned predictor — many duplicates, the bam discretization
    sweet spot."""
    bins = np.linspace(0, 10, n_unique)
    x_idx = RNG.integers(0, n_unique, n)
    x = bins[x_idx]
    y = np.sin(x) + 0.5 * np.cos(2 * x) + RNG.normal(0, 0.3, n)
    return x, y


def time_fit(engine_module, x, y, **kwargs):
    df = pd.DataFrame({"x": x, "y": y})
    best = float("inf")
    for _ in range(RUNS):
        t0 = time.perf_counter()
        gam = engine_module.Gam(predictors=["x"], target="y", family="gaussian", k=20, **kwargs)
        gam.fit(df, df["y"])
        dt = (time.perf_counter() - t0) * 1000.0
        if dt < best:
            best = dt
    return best


def main():
    if mr is None:
        print("mgcv_rust not installed")
        return
    print(f"Discretizable-data bench: best-of-{RUNS}, k=20, single smooth.\n")
    print(
        f"{'n':>9s}  {'unique':>7s}  {'gamrs ms':>9s}  {'mr_gam ms':>9s}  "
        f"{'mr_bam ms':>9s}  {'vs gam':>7s}  {'vs bam':>7s}"
    )
    print("-" * 80)
    for n in N_VALUES:
        for n_unique in N_UNIQUE_VALUES:
            if n_unique > n:
                continue
            x, y = synth_binned_gaussian(n, n_unique)
            g_ms = time_fit(gamrs, x, y)
            mr_gam_ms = time_fit(mr, x, y, discrete=False)
            mr_bam_ms = time_fit(mr, x, y, discrete=True)
            print(
                f"{n:>9d}  {n_unique:>7d}  {g_ms:>9.2f}  {mr_gam_ms:>9.2f}  "
                f"{mr_bam_ms:>9.2f}  {mr_gam_ms/g_ms:>6.2f}×  {mr_bam_ms/g_ms:>6.2f}×"
            )
        print()


if __name__ == "__main__":
    main()
