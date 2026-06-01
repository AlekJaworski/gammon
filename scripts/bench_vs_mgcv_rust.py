#!/usr/bin/env python
"""Performance + quality battery: gamrs.Gam vs mgcv_rust.Gam on mgcv-R fixtures.

Both gamrs and mgcv_rust expose a near-identical `Gam(predictors=..., target=...,
family=...)` API. This script runs the SAME fit on both engines using
mgcv-R-generated fixtures as the ground truth, captures wall-time per fit and
μ-prediction rel-err against mgcv R.

Usage:
    .venv/bin/python scripts/bench_vs_mgcv_rust.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import gamrs
import mgcv_rust as mr

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
RUNS = 5  # bench repeats per fixture


def load_xy(name: str):
    fx = json.loads((FIXTURES / f"{name}.json").read_text())
    inp = fx["inputs"]
    out = fx["mgcv_output"]
    xs = inp["x_train"]
    if isinstance(xs[0], list):
        x = np.asarray(xs, dtype=float)
    else:
        x = np.asarray(xs, dtype=float).reshape(-1, 1)
    y = np.asarray(inp["y_train"], dtype=float)
    mu_mgcv = np.asarray(out["predictions_train"], dtype=float)
    k = inp.get("k", [10])
    return x, y, mu_mgcv, k, inp


def max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1.0)))


def gam_kwargs(family: str, k: int, p_value: float | None, theta_init: float | None,
               engine: str):
    """family aliases differ between engines — normalise per-engine."""
    if engine == "mr":
        # mgcv_rust uses "binomial" / "inverse.gaussian" (mgcv R conventions).
        family = {"bernoulli": "binomial",
                  "inverse_gaussian": "inverse.gaussian"}.get(family, family)
    kwargs: dict = dict(family=family, k_default=k)
    if family == "negbin" and theta_init is not None:
        kwargs["negbin_theta"] = theta_init
    if family == "tweedie" and p_value is not None:
        kwargs["tweedie_p"] = p_value
    return kwargs


def fit_gamrs(x, y, family, k, p_value, theta_init):
    gam = gamrs.Gam(**gam_kwargs(family, k, p_value, theta_init, engine="gamrs"))
    gam.fit(x, y)
    return gam


def fit_mr(x, y, family, k, p_value, theta_init):
    gam = mr.Gam(**gam_kwargs(family, k, p_value, theta_init, engine="mr"))
    gam.fit(x, y)
    return gam


def predict_mu(gam, x, family) -> np.ndarray:
    return np.asarray(gam.predict(x, scale="response"))


def time_fit(builder, *args, runs: int = RUNS):
    """Warm-up + best-of-N timing. Returns (best_ms, last_gam)."""
    gam = builder(*args)
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        gam = builder(*args)
        dt = (time.perf_counter() - t0) * 1000.0
        if dt < best:
            best = dt
    return best, gam


# Fixture roster — pick representatives per family + scale tier.
SCENARIOS = [
    # (fixture_name, family, p_value, theta_init)
    ("1d_gaussian_smooth_n500_k10_cr",   "gaussian", None, None),
    ("1d_gaussian_smooth_n2000_k30_cr",  "gaussian", None, None),
    ("2d_gaussian_additive_n500_k10_cr", "gaussian", None, None),
    ("10d_gaussian_n3000_k8_cr",         "gaussian", None, None),
    ("1d_poisson_log_n300_k10_cr",       "poisson",  None, None),
    ("1d_bernoulli_logit_n1000_k10_cr",  "bernoulli", None, None),
    ("1d_nb_log_n300_k10_cr",            "negbin",   None, 5.0),
    ("2d_nb_log_n600_k8_cr",             "negbin",   None, 5.0),
    ("1d_gamma_log_n300_k10_cr",         "gamma",    None, None),
    ("1d_invgauss_log_n300_k10_cr",      "inverse_gaussian", None, None),
    ("1d_tweedie_log_n300_k10_cr",       "tweedie",  None, None),
]


def run():
    rows = []
    for name, family, p_val, th_init in SCENARIOS:
        try:
            x, y, mu_mgcv, k_list, _inp = load_xy(name)
        except FileNotFoundError:
            print(f"SKIP {name} — fixture missing")
            continue
        n, d = x.shape
        k = k_list[0]

        gt_ms = gt_rel = float("nan")
        gt_mu = None
        try:
            gt_ms, gt_gam = time_fit(fit_gamrs, x, y, family, k, p_val, th_init)
            gt_mu = predict_mu(gt_gam, x, family)
            gt_rel = max_rel(gt_mu, mu_mgcv)
        except Exception as e:
            print(f"gamrs FAIL on {name}: {type(e).__name__}: {e}")

        mr_ms = mr_rel = float("nan")
        mr_mu = None
        try:
            mr_ms, mr_gam = time_fit(fit_mr, x, y, family, k, p_val, th_init)
            mr_mu = predict_mu(mr_gam, x, family)
            mr_rel = max_rel(mr_mu, mu_mgcv)
        except Exception as e:
            print(f"mgcv_rust FAIL on {name}: {type(e).__name__}: {e}")

        cross = float("nan")
        if gt_mu is not None and mr_mu is not None:
            try:
                cross = max_rel(gt_mu, mr_mu)
            except Exception:
                pass

        speedup = (mr_ms / gt_ms) if (np.isfinite(gt_ms) and np.isfinite(mr_ms) and gt_ms > 0) else float("nan")
        rows.append({
            "fixture": name, "family": family, "n": n, "d": d,
            "gamrs_ms": gt_ms, "mr_ms": mr_ms, "speedup": speedup,
            "gamrs_rel_vs_R": gt_rel, "mr_rel_vs_R": mr_rel, "gamrs_vs_mr": cross,
        })

    print("\n=========  PERFORMANCE + QUALITY BATTERY: gamrs vs mgcv_rust  =========\n")
    print("| fixture | n | d | gamrs ms | mr ms | speedup | gamrs μ-rel vs R | mr μ-rel vs R | gamrs↔mr |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        gms = f"{r['gamrs_ms']:.2f}" if np.isfinite(r['gamrs_ms']) else "—"
        mms = f"{r['mr_ms']:.2f}" if np.isfinite(r['mr_ms']) else "—"
        sp  = f"{r['speedup']:.2f}×" if np.isfinite(r['speedup']) else "—"
        gr  = f"{r['gamrs_rel_vs_R']:.2e}" if np.isfinite(r['gamrs_rel_vs_R']) else "—"
        mr_ = f"{r['mr_rel_vs_R']:.2e}" if np.isfinite(r['mr_rel_vs_R']) else "—"
        cs  = f"{r['gamrs_vs_mr']:.2e}" if np.isfinite(r['gamrs_vs_mr']) else "—"
        print(f"| {r['fixture']} | {r['n']} | {r['d']} | {gms} | {mms} | {sp} | {gr} | {mr_} | {cs} |")
    print("\n(rel-err: max_i |μ̂_i − μ_mgcv_R_i| / (|μ_mgcv_R_i| + 1). Lower is better.)\n")


if __name__ == "__main__":
    run()
