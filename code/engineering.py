"""
Classic constrained mechanical-engineering design problems.

Three standard problems are implemented with their canonical formulations and
variable bounds, handled by a static penalty method (death-penalty-free: a
large quadratic penalty proportional to total constraint violation is added to
the objective). Each optimizer minimizes the penalized objective; we report the
best feasible design found over 30 independent runs and compare the best cost
to values reported in the metaheuristic literature.

Problems:
  1. Welded beam design        (4 vars, 5+ constraints)   target ~1.6952
  2. Pressure vessel design    (4 vars, 4 constraints)    target ~5885-6060
  3. Tension/compression spring(3 vars, 4 constraints)    target ~0.01266
"""
import numpy as np

PENALTY = 1e9


def _penalized(obj, g_list):
    viol = sum(max(0.0, gi) for gi in g_list)
    return obj + PENALTY * viol + PENALTY * (viol ** 2) if viol > 0 else obj


# ---------------------------------------------------------------------------
# 1. Welded beam
# ---------------------------------------------------------------------------
WELDED_BEAM = {
    "name": "Welded Beam",
    "D": 4,
    "lb": np.array([0.1, 0.1, 0.1, 0.1]),
    "ub": np.array([2.0, 10.0, 10.0, 2.0]),
    "ref": 1.724852,   # widely-cited best (Coello, GSA, GWO literature)
}

def welded_beam(x):
    x = np.atleast_2d(x)
    out = np.empty(x.shape[0])
    P = 6000.0; L = 14.0; E = 30e6; G = 12e6
    tau_max = 13600.0; sigma_max = 30000.0; delta_max = 0.25
    for k in range(x.shape[0]):
        h, l, t, b = x[k]
        h = max(h, 1e-6); l = max(l, 1e-6); t = max(t, 1e-6); b = max(b, 1e-6)
        cost = 1.10471 * h ** 2 * l + 0.04811 * t * b * (14.0 + l)
        M = P * (L + l / 2.0)
        R = np.sqrt(l ** 2 / 4.0 + ((h + t) / 2.0) ** 2)
        J = 2.0 * (np.sqrt(2.0) * h * l * (l ** 2 / 12.0 + ((h + t) / 2.0) ** 2))
        tau1 = P / (np.sqrt(2.0) * h * l)
        tau2 = M * R / J
        tau = np.sqrt(tau1 ** 2 + 2.0 * tau1 * tau2 * l / (2.0 * R) + tau2 ** 2)
        sigma = 6.0 * P * L / (b * t ** 2)
        delta = 6.0 * P * L ** 3 / (E * t ** 3 * b)
        Pc = (4.013 * E * np.sqrt(t ** 2 * b ** 6 / 36.0) / L ** 2) * (1.0 - t / (2.0 * L) * np.sqrt(E / (4.0 * G)))
        g = [tau - tau_max,
             sigma - sigma_max,
             h - b,
             0.10471 * h ** 2 + 0.04811 * t * b * (14.0 + l) - 5.0,
             0.125 - h,
             delta - delta_max,
             P - Pc]
        out[k] = _penalized(cost, g)
    return out if out.shape[0] > 1 else float(out[0])

def welded_beam_feasible(x):
    h, l, t, b = x
    P = 6000.0; L = 14.0; E = 30e6; G = 12e6
    tau_max = 13600.0; sigma_max = 30000.0; delta_max = 0.25
    M = P * (L + l / 2.0); R = np.sqrt(l ** 2 / 4.0 + ((h + t) / 2.0) ** 2)
    J = 2.0 * (np.sqrt(2.0) * h * l * (l ** 2 / 12.0 + ((h + t) / 2.0) ** 2))
    tau1 = P / (np.sqrt(2.0) * h * l); tau2 = M * R / J
    tau = np.sqrt(tau1 ** 2 + 2.0 * tau1 * tau2 * l / (2.0 * R) + tau2 ** 2)
    sigma = 6.0 * P * L / (b * t ** 2); delta = 6.0 * P * L ** 3 / (E * t ** 3 * b)
    Pc = (4.013 * E * np.sqrt(t ** 2 * b ** 6 / 36.0) / L ** 2) * (1.0 - t / (2.0 * L) * np.sqrt(E / (4.0 * G)))
    g = [tau - tau_max, sigma - sigma_max, h - b,
         0.10471 * h ** 2 + 0.04811 * t * b * (14.0 + l) - 5.0,
         0.125 - h, delta - delta_max, P - Pc]
    return all(gi <= 1e-4 for gi in g)


# ---------------------------------------------------------------------------
# 2. Pressure vessel (continuous-relaxed; Ts,Th multiples of 0.0625 in lit.)
# ---------------------------------------------------------------------------
PRESSURE_VESSEL = {
    "name": "Pressure Vessel",
    "D": 4,
    "lb": np.array([0.0625, 0.0625, 10.0, 10.0]),
    "ub": np.array([6.1875, 6.1875, 200.0, 200.0]),
    "ref": 6059.714,
}

def pressure_vessel(x):
    x = np.atleast_2d(x)
    out = np.empty(x.shape[0])
    for k in range(x.shape[0]):
        Ts, Th, R, L = x[k]
        # discretize thickness to multiples of 0.0625 as in standard problem
        Ts = round(Ts / 0.0625) * 0.0625
        Th = round(Th / 0.0625) * 0.0625
        Ts = max(Ts, 0.0625); Th = max(Th, 0.0625)
        cost = (0.6224 * Ts * R * L + 1.7781 * Th * R ** 2 +
                3.1661 * Ts ** 2 * L + 19.84 * Ts ** 2 * R)
        g = [-Ts + 0.0193 * R,
             -Th + 0.00954 * R,
             -np.pi * R ** 2 * L - (4.0 / 3.0) * np.pi * R ** 3 + 1296000.0,
             L - 240.0]
        out[k] = _penalized(cost, g)
    return out if out.shape[0] > 1 else float(out[0])

def pressure_vessel_feasible(x):
    Ts, Th, R, L = x
    Ts = round(Ts / 0.0625) * 0.0625; Th = round(Th / 0.0625) * 0.0625
    g = [-Ts + 0.0193 * R, -Th + 0.00954 * R,
         -np.pi * R ** 2 * L - (4.0 / 3.0) * np.pi * R ** 3 + 1296000.0, L - 240.0]
    return all(gi <= 1e-2 for gi in g)


# ---------------------------------------------------------------------------
# 3. Tension / compression spring
# ---------------------------------------------------------------------------
SPRING = {
    "name": "Tension/Compression Spring",
    "D": 3,
    "lb": np.array([0.05, 0.25, 2.0]),
    "ub": np.array([2.0, 1.3, 15.0]),
    "ref": 0.012665,
}

def spring(x):
    x = np.atleast_2d(x)
    out = np.empty(x.shape[0])
    for k in range(x.shape[0]):
        d, D, N = x[k]
        d = max(d, 1e-6); D = max(D, 1e-6)
        cost = (N + 2.0) * D * d ** 2
        g = [1.0 - (D ** 3 * N) / (71785.0 * d ** 4),
             (4.0 * D ** 2 - d * D) / (12566.0 * (D * d ** 3 - d ** 4)) + 1.0 / (5108.0 * d ** 2) - 1.0,
             1.0 - 140.45 * d / (D ** 2 * N),
             (D + d) / 1.5 - 1.0]
        out[k] = _penalized(cost, g)
    return out if out.shape[0] > 1 else float(out[0])

def spring_feasible(x):
    d, D, N = x
    g = [1.0 - (D ** 3 * N) / (71785.0 * d ** 4),
         (4.0 * D ** 2 - d * D) / (12566.0 * (D * d ** 3 - d ** 4)) + 1.0 / (5108.0 * d ** 2) - 1.0,
         1.0 - 140.45 * d / (D ** 2 * N), (D + d) / 1.5 - 1.0]
    return all(gi <= 1e-4 for gi in g)


PROBLEMS = [
    (WELDED_BEAM, welded_beam, welded_beam_feasible),
    (PRESSURE_VESSEL, pressure_vessel, pressure_vessel_feasible),
    (SPRING, spring, spring_feasible),
]
