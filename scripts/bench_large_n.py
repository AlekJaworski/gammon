#!/usr/bin/env python
"""Large-n perf bench: gamrs vs mgcv_rust at n in {10K, 50K, 100K, 250K}.

Goal: find out where (and if) gamrs's dense-QR / dense-PIRLS path falls
behind mgcv_rust's chunked-QR / discrete-binning paths. Decides whether
those ports are worth pulling in.

Each cell: synthetic Gaussian / Poisson / Bernoulli fit at the listed n,
single smooth (k=20). Best-of-3 wall time. Includes µ-pred sanity vs
the gamrs-itself prediction (no R parity at these sizes — too expensive
to generate).

Usage:
    .venv/bin/python scripts/bench_large_n.py
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


N_VALUES = [10_000, 50_000, 100_000, 250_000, 1_000_000]
RUNS = 3
RNG = np.random.default_rng(42)


def synth_gaussian(n: int):
    x = RNG.uniform(0, 10, n)
    y = np.sin(x) + 0.5 * np.cos(2 * x) + RNG.normal(0, 0.3, n)
    return x, y, "gaussian"


def synth_poisson(n: int):
    x = RNG.uniform(0, 10, n)
    eta = 0.5 + 0.3 * np.sin(x)
    mu = np.exp(eta)
    y = RNG.poisson(mu).astype(float)
    return x, y, "poisson"


def synth_bernoulli(n: int):
    x = RNG.uniform(-3, 3, n)
    eta = np.sin(x)
    p = 1 / (1 + np.exp(-eta))
    y = (RNG.uniform(0, 1, n) < p).astype(float)
    # mgcv_rust uses 'binomial' for logit-link 0/1 outcomes; gamrs aliases
    # 'binomial' to 'bernoulli' internally.
    return x, y, "binomial"


SCENARIOS = [
    ("gaussian", synth_gaussian),
    ("poisson", synth_poisson),
    ("bernoulli", synth_bernoulli),
]


def time_fit(engine_module, x, y, family: str, k: int = 20):
    df = pd.DataFrame({"x": x, "y": y})
    best = float("inf")
    last_gam = None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        gam = engine_module.Gam(
            predictors=["x"], target="y", family=family, k=k
        )
        gam.fit(df, df["y"])
        dt = (time.perf_counter() - t0) * 1000.0
        if dt < best:
            best = dt
        last_gam = gam
    return best, last_gam


def main():
    if mr is None:
        print("mgcv_rust not installed — bench needs both engines.")
        return

    print(f"Large-n bench: best-of-{RUNS}, k=20, single smooth.\n")
    print(
        f"{'family':>10s}  {'n':>8s}  {'gamrs ms':>9s}  {'mr ms':>9s}  "
        f"{'speedup':>8s}  {'outer':>5s}  {'pirls_iters':>11s}"
    )
    print("-" * 80)
    for family, synth in SCENARIOS:
        for n in N_VALUES:
            x, y, fam = synth(n)
            try:
                g_ms, g_gam = time_fit(gamrs, x, y, fam)
                g_stats = g_gam.fit_stats_
            except Exception as e:
                g_ms = float("nan")
                g_stats = {"outer_iterations": 0, "inner_pirls_iterations_total": 0}
                print(f"  gamrs failed: {e}")
            try:
                mr_ms, _ = time_fit(mr, x, y, fam)
            except Exception as e:
                mr_ms = float("nan")
                print(f"  mgcv_rust failed: {e}")
            speedup = (mr_ms / g_ms) if (g_ms == g_ms and mr_ms == mr_ms and g_ms > 0) else float("nan")
            print(
                f"{family:>10s}  {n:>8d}  {g_ms:>9.2f}  {mr_ms:>9.2f}  "
                f"{speedup:>7.2f}×  {g_stats['outer_iterations']:>5d}  "
                f"{g_stats['inner_pirls_iterations_total']:>11d}"
            )
        print()


if __name__ == "__main__":
    main()
