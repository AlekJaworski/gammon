#!/usr/bin/env python
"""Sweep outer-Newton (grad_tol, reml_tol) per fixture.

For each fixture × tolerance combo, capture:
  - wall-time
  - outer iters
  - inner PIRLS calls + iters
  - rho_hat drift vs the tightest-tolerance run (parity sentinel)
  - μ-prediction parity vs mgcv R reference

Goal: pick the loosest (grad_tol, reml_tol) per family that doesn't move
rho_hat by more than ~1e-3 or μ-rel-err by more than ~1e-4 vs the tight
baseline. Lock the pick into `Loss::outer_tuning()` per family.

Usage:
    .venv/bin/python scripts/sweep_tolerances.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import gamrs
from gamrs._gamrs_native import set_outer_tuning_override

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# Sweep grid: tight → loose. (grad_tol, reml_tol). Pinned at one
# representative ratio for cleaner reading; could grid-2-d if needed.
TOL_GRID = [
    ("tight", 1e-9, 1e-10),
    ("mgcv*", 5e-7, 1e-7),   # current default
    ("loose1", 5e-6, 1e-6),
    ("loose2", 5e-5, 1e-5),
    ("loose3", 5e-4, 1e-4),
]

FIXTURE_NAMES = [
    "1d_gaussian_smooth_n2000_k30_cr",
    "10d_gaussian_n3000_k8_cr",
    "1d_poisson_log_n300_k10_cr",
    "1d_bernoulli_logit_n1000_k10_cr",
    "1d_nb_log_n300_k10_cr",
    "2d_nb_log_n600_k8_cr",
    "1d_gamma_log_n300_k10_cr",
    "1d_invgauss_log_n300_k10_cr",
    "1d_tweedie_log_n300_k10_cr",
]

FAMILY_MAP = {"nb": "negbin", "inverse_gaussian": "inverse_gaussian"}
RUNS = 3  # median of N to dampen noise


def load_fixture(name: str):
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


def run_one(name: str, grad_tol: float, reml_tol: float):
    df, predictors, fam = load_fixture(name)
    set_outer_tuning_override(grad_tol=grad_tol, reml_tol=reml_tol)
    elapsed = []
    rho = None
    stats = None
    mu_pred = None
    for _ in range(RUNS):
        g = gamrs.Gam(predictors=predictors, target="y", family=fam)
        t0 = time.perf_counter_ns()
        g.fit(df, df["y"])
        elapsed.append((time.perf_counter_ns() - t0) / 1e6)
        rho = np.asarray(g.rho_)
        stats = g.fit_stats_
        # μ prediction on training x (for parity sentinel; OOS uses test)
        eta = g.predict(df)
        mu_pred = np.asarray(eta) if fam == "gaussian" else np.exp(np.asarray(eta))
    set_outer_tuning_override(grad_tol=None, reml_tol=None)
    return {
        "ms": float(np.median(elapsed)),
        "rho": rho,
        "mu": mu_pred,
        "outer": stats["outer_iterations"],
        "pirls_calls": stats["inner_pirls_calls"],
        "pirls_iters": stats["inner_pirls_iterations_total"],
        "ls_trials": stats["line_search_trials"],
    }


def main():
    print("Tolerance sweep — picking loosest (grad_tol, reml_tol) per family.")
    print(f"  - {RUNS} runs/combo; median ms reported.")
    print(f"  - rho_drift = max_i |ρ̂[i] − ρ̂_tight[i]|.")
    print(f"  - mu_drift  = max_i |μ̂[i] − μ̂_tight[i]| / (|μ̂_tight[i]| + 1).")
    print()
    for name in FIXTURE_NAMES:
        print(f"=== {name} ===")
        tight = run_one(name, *TOL_GRID[0][1:])
        rho_tight = tight["rho"]
        mu_tight = tight["mu"]
        print(
            f"  {'label':>7s}  {'grad':>6s}  {'reml':>6s}  "
            f"{'ms':>7s}  {'speedup':>7s}  {'outer':>5s}  {'p_calls':>7s}  "
            f"{'rho_drift':>9s}  {'mu_drift':>8s}"
        )
        for label, gt, rt in TOL_GRID:
            r = run_one(name, gt, rt)
            rho_drift = float(np.max(np.abs(r["rho"] - rho_tight)))
            denom = np.abs(mu_tight) + 1.0
            mu_drift = float(np.max(np.abs(r["mu"] - mu_tight) / denom))
            speedup = tight["ms"] / r["ms"] if r["ms"] > 0 else float("inf")
            print(
                f"  {label:>7s}  {gt:6.0e}  {rt:6.0e}  "
                f"{r['ms']:7.2f}  {speedup:6.2f}×  "
                f"{r['outer']:5d}  {r['pirls_calls']:7d}  "
                f"{rho_drift:9.2e}  {mu_drift:8.2e}"
            )
        print()


if __name__ == "__main__":
    main()
