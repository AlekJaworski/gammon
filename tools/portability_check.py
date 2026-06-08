"""Release portability check — exercises the bundled OpenBLAS via a real fit.

Background: the 0.11.0–0.11.2 Linux wheels SIGILLed on AVX2-only hosts because
the statically-linked OpenBLAS baked the CI runner's AVX-512 kernels instead of
being runtime-CPU-dispatched. `OPENBLAS_DYNAMIC_ARCH=1` (set in release.yml)
fixes that by compiling every x86 kernel and selecting one at runtime via
cpuid.

release.yml runs this under Intel SDE emulating a Haswell (AVX2-only) CPU
(`sde64 -hsw -- python tools/portability_check.py`). A real GAM fit routes its
matmuls through OpenBLAS GEMM (via `ndarray/blas`), so if any executed code —
kernel or OpenBLAS common/driver code — uses an instruction outside the Haswell
ISA (e.g. an AVX-512 `vbroadcastsd …%zmm` baked in by an AVX-512 build runner),
SDE's chip-check aborts with a nonzero exit and fails the release. Pass = the
fit completes and prints the line below. See the guard step's comment for why
emulation (not a forced OPENBLAS_CORETYPE on the AVX-512 runner) is required.
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
