"""
Shifted-and-rotated single-objective bound-constrained benchmark suite (F1-F10).

This is a self-contained, dependency-free benchmark suite built from the
classical basic test-function families used in the CEC competitions
(Bent Cigar, Zakharov, High-Conditioned Elliptic, Discus -- unimodal;
Rosenbrock, Rastrigin, Schaffer F7, Levy, Ackley, Griewank -- multimodal).
Each function is shifted by a fixed vector (so the global optimum is interior
and not at the origin) and rotated by a fixed orthonormal matrix (so the
landscape is non-separable). The shift vectors and rotation matrices are
generated deterministically from fixed seeds and REUSED across every run of
every optimizer, so all algorithms are compared on identical landscapes.
The search domain is [-100, 100]^D and the known global optimum value of
function k is its bias (100*k).

Design references for the basic functions and the shift/rotate methodology:
  N.H. Awad, M.Z. Ali, J.J. Liang, B.Y. Qu, P.N. Suganthan, "Problem
  Definitions and Evaluation Criteria for the CEC 2017 Special Session on
  Single Objective Real-Parameter Numerical Optimization", Tech. Rep., 2016.
"""
import numpy as np

DOMAIN = (-100.0, 100.0)

# ---------------------------------------------------------------------------
# Basic functions (operate on already shifted/rotated z)
# ---------------------------------------------------------------------------

def _bent_cigar(z):
    return z[..., 0] ** 2 + 1e6 * np.sum(z[..., 1:] ** 2, axis=-1)

def _sphere(z):
    return np.sum(z ** 2, axis=-1)

def _zakharov(z):
    s2 = np.sum(z ** 2, axis=-1)
    idx = np.arange(1, z.shape[-1] + 1)
    s3 = np.sum(0.5 * idx * z, axis=-1)
    return s2 + s3 ** 2 + s3 ** 4

def _rosenbrock(z):
    z = z * 2.048 / 100.0 + 1.0
    a = z[..., :-1]
    b = z[..., 1:]
    return np.sum(100.0 * (a ** 2 - b) ** 2 + (a - 1.0) ** 2, axis=-1)

def _rastrigin(z):
    z = z * 5.12 / 100.0
    return np.sum(z ** 2 - 10.0 * np.cos(2 * np.pi * z) + 10.0, axis=-1)

def _griewank(z):
    z = z * 600.0 / 100.0
    s = np.sum(z ** 2, axis=-1) / 4000.0
    idx = np.sqrt(np.arange(1, z.shape[-1] + 1))
    p = np.prod(np.cos(z / idx), axis=-1)
    return s - p + 1.0

def _ackley(z):
    D = z.shape[-1]
    s1 = np.sqrt(np.sum(z ** 2, axis=-1) / D)
    s2 = np.sum(np.cos(2 * np.pi * z), axis=-1) / D
    return -20.0 * np.exp(-0.2 * s1) - np.exp(s2) + 20.0 + np.e

def _discus(z):
    return 1e6 * z[..., 0] ** 2 + np.sum(z[..., 1:] ** 2, axis=-1)

def _levy(z):
    z = z / 10.0
    w = 1.0 + (z) / 4.0
    term1 = np.sin(np.pi * w[..., 0]) ** 2
    wi = w[..., :-1]
    term2 = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2), axis=-1)
    term3 = (w[..., -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[..., -1]) ** 2)
    return term1 + term2 + term3

def _high_cond_elliptic(z):
    D = z.shape[-1]
    idx = np.arange(D)
    coef = 1e6 ** (idx / (D - 1))
    return np.sum(coef * z ** 2, axis=-1)

def _schaffer_f7(z):
    z = z / 5.0
    si = np.sqrt(z[..., :-1] ** 2 + z[..., 1:] ** 2)
    D = z.shape[-1]
    t = si ** 0.5 * (np.sin(50.0 * si ** 0.2) + 1.0)
    return (np.sum(t, axis=-1) / (D - 1)) ** 2

# ---------------------------------------------------------------------------
# Function registry: each entry = (name, class, basic-fn, bias)
# class in {unimodal, multimodal, hybrid, composition}
# ---------------------------------------------------------------------------

_BASE = [
    ("F1",  "unimodal",   _bent_cigar,         100),
    ("F2",  "unimodal",   _zakharov,           200),
    ("F3",  "unimodal",   _high_cond_elliptic, 300),
    ("F4",  "multimodal", _rosenbrock,         400),
    ("F5",  "multimodal", _rastrigin,          500),
    ("F6",  "multimodal", _schaffer_f7,        600),
    ("F7",  "multimodal", _levy,               700),
    ("F8",  "multimodal", _ackley,             800),
    ("F9",  "multimodal", _griewank,           900),
    ("F10", "unimodal",   _discus,            1000),
]


class CEC2017Function:
    def __init__(self, name, cls, fn, bias, D, seed):
        self.name = name
        self.cls = cls
        self._fn = fn
        self.bias = bias
        self.D = D
        self.lb = DOMAIN[0]
        self.ub = DOMAIN[1]
        rng = np.random.default_rng(seed)
        # fixed shift inside [-80,80] so optimum is interior
        self.shift = rng.uniform(-80, 80, size=D)
        # fixed orthonormal rotation matrix via QR
        A = rng.standard_normal((D, D))
        Q, _ = np.linalg.qr(A)
        self.rot = Q
        self.fopt = bias  # global optimum value

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        z = (x - self.shift) @ self.rot.T
        val = self._fn(z) + self.bias
        return val if val.shape[0] > 1 else float(val[0])


def get_suite(D, seed_base=20240617):
    """Return list of CEC2017Function objects for given dimension D."""
    funcs = []
    for i, (name, cls, fn, bias) in enumerate(_BASE):
        funcs.append(CEC2017Function(name, cls, fn, bias, D, seed_base + i))
    return funcs


if __name__ == "__main__":
    suite = get_suite(10)
    for f in suite:
        x = f.shift.reshape(1, -1)  # evaluate at optimum
        print(f"{f.name} ({f.cls}): f(x*)={f(x):.4f}  bias={f.bias}")
