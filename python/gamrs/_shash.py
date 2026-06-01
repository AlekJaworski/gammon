"""Sinh-ArcSinh (SHASH) distribution helpers for qgam's err / co parameter.

This is a faithful port of mgcv_rust's :mod:`mgcv_rust._shash` (itself a
port of R qgam's ``.fitShash`` / ``.llkShash`` / ``.shashCDF`` /
``.shashQf`` / ``.shashMode`` and ``.getErrParam``). It exists so gamrs's
quantile ``fast_oos`` path can pick the ELF likelihood scale from a
qgam-faithful SHASH fit of the standardised pilot residuals rather than
the crude ``err = 0.05`` constant baked into the Rust σ heuristic.

Parameterisation: ``(mu, tau, eps, phi)`` with ``sigma = exp(tau)`` and
``delta = exp(phi)``. All log-likelihood quantities follow R's sign
conventions so the values match mgcv_rust / qgam directly.

Implemented with numpy + scipy only (no new heavy deps); the only scipy
surface used is :func:`scipy.optimize.minimize` /
:func:`scipy.optimize.minimize_scalar` and the normal CDF / quantile
(:func:`scipy.special.ndtr` / :func:`scipy.special.ndtri`).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import ndtr, ndtri


def _shash_z_and_parts(x: np.ndarray, mu: float, tau: float, eps: float, phi: float):
    """Shared intermediate quantities used by loglik and gradient."""
    sig = np.exp(tau)
    del_ = np.exp(phi)
    z = (x - mu) / (sig * del_)
    asinhZ = np.arcsinh(z)
    dTasMe = del_ * asinhZ - eps  # R's dTasMe
    g = -dTasMe  # R's g
    SS = np.sinh(dTasMe)
    CC = np.cosh(dTasMe)
    sSp1 = np.sqrt(z**2 + 1.0)  # sqrt(z^2+1), R's sSp1
    return sig, del_, z, asinhZ, dTasMe, g, SS, CC, sSp1


def shash_loglik(x: np.ndarray, mu: float, tau: float, eps: float, phi: float) -> float:
    """Sum of SHASH log-densities over array x.  Mirrors R's .llkShash(deriv=0)$l0."""
    x = np.asarray(x, dtype=float)
    _, _, z, _, dTasMe, _, SS, CC, _ = _shash_z_and_parts(x, mu, tau, eps, phi)
    return float(
        np.sum(
            -tau
            - 0.5 * np.log(2.0 * np.pi)
            + np.log(CC)
            - 0.5 * np.log1p(z**2)
            - 0.5 * SS**2
        )
    )


def shash_loglik_grad(
    x: np.ndarray, mu: float, tau: float, eps: float, phi: float
) -> np.ndarray:
    """Gradient of sum(log L) w.r.t. (mu, tau, eps, phi).

    Mirrors R's .llkShash(deriv=1)$l1 = c(Dm, Dt, De, Dp).
    """
    x = np.asarray(x, dtype=float)
    sig, del_, z, asinhZ, _, g, _, _, sSp1 = _shash_z_and_parts(x, mu, tau, eps, phi)
    zsd = z * sig * del_  # = x - mu

    De = np.tanh(g) - 0.5 * np.sinh(2.0 * g)
    Dm = (1.0 / (del_ * sig * sSp1)) * (del_ * De + z / sSp1)
    Dt = zsd * Dm - 1.0
    Dp = Dt + 1.0 - del_ * asinhZ * De

    return np.array(
        [float(np.sum(Dm)), float(np.sum(Dt)), float(np.sum(De)), float(np.sum(Dp))]
    )


def fit_shash(r: np.ndarray) -> np.ndarray:
    """Fit SHASH to data r via BFGS.  Mirrors R's .fitShash.

    Returns array ``[mu, tau, eps, phi]`` at the MLE. Raises on BFGS
    divergence / non-finite result so the caller can fall back to the
    documented ``err = 0.05`` default.
    """
    r = np.asarray(r, dtype=float)
    std = float(np.std(r))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("SHASH fit: degenerate residuals (zero / non-finite std)")
    init = np.array([float(np.mean(r)), float(np.log(std)), 0.0, 0.0])

    def neg_ll(params: np.ndarray) -> float:
        return -shash_loglik(r, *params)

    def neg_grad(params: np.ndarray) -> np.ndarray:
        return -shash_loglik_grad(r, *params)

    res = minimize(neg_ll, init, jac=neg_grad, method="BFGS")
    if not np.all(np.isfinite(res.x)):
        raise ValueError("SHASH fit: BFGS produced non-finite parameters")
    return res.x


def shash_qf(p: float, params: np.ndarray) -> float:
    """SHASH quantile function (inverse CDF).  Mirrors R's .shashQf."""
    mu, tau, eps, phi = params
    sig = np.exp(tau)
    del_ = np.exp(phi)
    return float(
        mu + (del_ * sig) * np.sinh((1.0 / del_) * np.arcsinh(ndtri(p)) + eps / del_)
    )


def shash_cdf(x: float, params: np.ndarray) -> float:
    """SHASH CDF.  Mirrors R's .shashCDF."""
    mu, tau, eps, phi = params
    sig = np.exp(tau)
    del_ = np.exp(phi)
    return float(ndtr(np.sinh((np.arcsinh((x - mu) / (del_ * sig)) - eps / del_) * del_)))


def shash_mode(params: np.ndarray) -> float:
    """Mode of SHASH.  Mirrors R's .shashMode (optimise over .shashQf range)."""
    lo = shash_qf(0.001, params)
    hi = shash_qf(0.999, params)
    mu, tau, eps, phi = params

    def neg_loglik(x_scalar: float) -> float:
        return -shash_loglik(np.array([x_scalar]), mu, tau, eps, phi)

    res = minimize_scalar(neg_loglik, bounds=(lo, hi), method="bounded")
    return float(res.x)


def _dm_at_point(x_scalar: float, mu: float, tau: float, eps: float, phi: float) -> float:
    """d(log f)/d(mu) at a single point x_scalar.  Used for the bandwidth formula."""
    x = np.array([x_scalar])
    sig, del_, z, _, _, g, _, _, sSp1 = _shash_z_and_parts(x, mu, tau, eps, phi)
    De = np.tanh(g) - 0.5 * np.sinh(2.0 * g)
    Dm = (1.0 / (del_ * sig * sSp1)) * (del_ * De + z / sSp1)
    return float(Dm[0])


def compute_err_param(r: np.ndarray, d: float, qu_list: list[float]) -> list[float]:
    """Replicate qgam:::.getErrParam (port of mgcv_rust._shash.compute_err_param).

    Args:
        r:       Standardised residuals ``(y - fitted) / sqrt(varHat)``.
        d:       Effective df (parametric df + sum of smooth EDFs).
        qu_list: Quantile levels.

    Returns:
        ``err`` values (= qgam's bandwidth-derived ``co`` scaling) for each
        quantile. Each is clipped to ``[*, 1.0]`` exactly as in qgam.

    Raises:
        Exception: if the SHASH BFGS fit diverges; the caller is expected to
        fall back to ``err = 0.05``.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    par_sh = fit_shash(r)
    pmode_x = shash_mode(par_sh)
    pmode = shash_cdf(pmode_x, par_sh)

    errs: list[float] = []
    for qu in qu_list:
        quX = float(qu)
        if abs(quX - pmode) < 0.05:
            quX = pmode + np.sign(quX - pmode) * 0.05
            quX = float(np.clip(quX, 0.01, 0.99))
        qhat = shash_qf(quX, par_sh)
        mu_, tau_, eps_, phi_ = par_sh
        lf0 = shash_loglik(np.array([qhat]), mu_, tau_, eps_, phi_)
        # R: lf1_raw <- -.llkShash(qhat, ...)$l1[1]  => negative of Dm at qhat
        lf1_raw = -_dm_at_point(qhat, mu_, tau_, eps_, phi_)
        lf1 = float(np.log(abs(lf1_raw)) + lf0)
        h = (d * 9.0 / (n * np.pi**4)) ** (1.0 / 3.0) * np.exp(
            lf0 / 3.0 - 2.0 * lf1 / 3.0
        )
        errs.append(float(min(h * 2.0 * np.log(2.0) / np.sqrt(2.0 * np.pi), 1.0)))
    return errs
