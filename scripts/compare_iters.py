#!/usr/bin/env python
"""Compare per-fit iteration counts: gamrs FitStats vs mgcv_rust internals.

Goal: make the perf gap legible. For each fixture, dump the gamrs stats
(outer iters, line-search trials, PIRLS calls + iters, NoRefresh hits)
alongside mgcv_rust's equivalents where available. Where mgcv_rust doesn't
expose a counter, the column is left blank — the gamrs side alone is
enough to triage iter-count vs per-eval gaps.

Usage:
    .venv/bin/python scripts/compare_iters.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import gamrs

try:
    import mgcv_rust as mr
except ImportError:
    mr = None

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

FIXTURE_NAMES = [
    "1d_gaussian_smooth_n500_k10_cr",
    "1d_gaussian_smooth_n2000_k30_cr",
    "2d_gaussian_additive_n500_k10_cr",
    "10d_gaussian_n3000_k8_cr",
    "1d_poisson_log_n300_k10_cr",
    "1d_bernoulli_logit_n1000_k10_cr",
    "1d_nb_log_n300_k10_cr",
    "2d_nb_log_n600_k8_cr",
    "1d_gamma_log_n300_k10_cr",
    "1d_invgauss_log_n300_k10_cr",
    "1d_tweedie_log_n300_k10_cr",
]

FAMILY_MAP = {
    "nb": "negbin",
    "inverse_gaussian": "inverse_gaussian",
}


def fixture_inputs(name: str):
    fx = json.loads((FIXTURES / f"{name}.json").read_text())
    inp = fx["inputs"]
    X = np.asarray(inp["x_train"], dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    elif X.ndim == 2 and X.shape[0] < X.shape[1]:
        X = X.T
    y = np.asarray(inp["y_train"], dtype=float)
    fam = inp["family"].lower().replace(" ", "_").replace(".", "_")
    fam = FAMILY_MAP.get(fam, fam)
    df = {f"x{i}": X[:, i] for i in range(X.shape[1])}
    df["y"] = y
    return pd.DataFrame(df), [f"x{i}" for i in range(X.shape[1])], fam


def gamrs_run(name: str):
    df, predictors, fam = fixture_inputs(name)
    g = gamrs.Gam(predictors=predictors, target="y", family=fam)
    t0 = time.perf_counter_ns()
    g.fit(df, df["y"])
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    return g.fit_stats_, elapsed_ms


def mr_run(name: str):
    if mr is None:
        return None, None
    df, predictors, fam = fixture_inputs(name)
    try:
        g = mr.Gam(predictors=predictors, target="y", family=fam)
        t0 = time.perf_counter_ns()
        g.fit(df, df["y"])
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    except Exception as e:
        return f"err: {e.__class__.__name__}", None
    n_iters = getattr(g, "n_iters_", None)
    return {"n_iters": n_iters}, elapsed_ms


def main():
    print("=" * 116)
    print(
        f"{'fixture':36s}  "
        f"{'gamrs ms':>9s}  {'mr ms':>7s}  {'ratio':>6s}  "
        f"{'outer':>5s}  {'ls':>4s}  {'pirls_c':>7s}  {'pirls_i':>7s}  "
        f"{'iters/call':>10s}  {'NR hits':>9s}  {'mr_outer':>8s}"
    )
    print("-" * 116)
    for name in FIXTURE_NAMES:
        stats, elapsed = gamrs_run(name)
        mr_stats, mr_elapsed = mr_run(name)
        mr_ms = f"{mr_elapsed:7.2f}" if mr_elapsed else "    n/a"
        ratio = (elapsed / mr_elapsed) if mr_elapsed else float("nan")
        ratio_s = f"{ratio:5.2f}×" if mr_elapsed else "   -  "
        mr_outer = mr_stats.get("n_iters") if isinstance(mr_stats, dict) else None
        nr = f"{stats['no_refresh_hits']}/{stats['no_refresh_attempts']}"
        print(
            f"{name:36s}  "
            f"{elapsed:9.2f}  {mr_ms}  {ratio_s}  "
            f"{stats['outer_iterations']:5d}  "
            f"{stats['line_search_trials']:4d}  "
            f"{stats['inner_pirls_calls']:7d}  "
            f"{stats['inner_pirls_iterations_total']:7d}  "
            f"{stats['pirls_iters_per_call']:10.1f}  "
            f"{nr:>9s}  "
            f"{(str(mr_outer) if mr_outer is not None else '-'):>8s}"
        )
    print()
    print("Reading the gap:")
    print("  • ratio > 1: gamrs slower than mgcv_rust")
    print("  • outer much higher than mr_outer: convergence/Hessian-quality gap")
    print("  • pirls_calls × pirls_iters dominates: inner cost gap (try NoRefresh)")
    print("  • iters/call high: per-call PIRLS convergence (tolerance/Newton-Fisher)")


if __name__ == "__main__":
    main()
