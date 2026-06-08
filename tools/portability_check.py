"""Release portability check — exercises the bundled OpenBLAS via a real fit.

Background: the 0.11.0–0.11.2 Linux wheels SIGILLed on AVX2-only hosts because
the statically-linked OpenBLAS baked the CI runner's AVX-512 kernels instead of
being runtime-CPU-dispatched. `OPENBLAS_DYNAMIC_ARCH=1` (set in release.yml)
fixes that by compiling every x86 kernel and selecting one at runtime via
cpuid.

This script does NOT decide pass/fail on its own — release.yml runs it under
`OPENBLAS_VERBOSE=2 OPENBLAS_CORETYPE=HASWELL` and greps the OpenBLAS init line
on stderr. A real GAM fit forces a BLAS call, so OpenBLAS prints which kernel it
selected. Forcing a non-native (AVX2-only Haswell) kernel only takes effect in a
DYNAMIC_ARCH build; a single-arch AVX-512 build ignores `OPENBLAS_CORETYPE` and
keeps reporting its baked core, which the workflow rejects. If the wheel cannot
run the forced kernel at all, the SIGILL surfaces here as a nonzero exit.
"""

import numpy as np

import gamrs

rng = np.random.default_rng(0)
x = np.linspace(0.0, 1.0, 400)
y = np.sin(2.0 * np.pi * x) + 0.1 * rng.standard_normal(400)

# A penalized CR fit goes through Array2::dot + Cholesky/solve, i.e. OpenBLAS.
g = gamrs.GAM("gaussian", k=10).fit(x, y)
pred = np.asarray(g.predict(x), dtype=float)
rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
assert np.isfinite(rmse) and rmse < 1.0, f"implausible fit (rmse={rmse})"
print(f"portability_check: gaussian fit OK, rmse={rmse:.4f}, n={x.size}")
