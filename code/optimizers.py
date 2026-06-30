"""
Population-based metaheuristic optimizers used in this study.

Implemented from scratch with NumPy:
  - GWO   : canonical Grey Wolf Optimizer (Mirjalili et al., 2014)
  - PSO   : standard Particle Swarm Optimization with inertia weight
  - DE    : Differential Evolution (DE/rand/1/bin)
  - WOA   : Whale Optimization Algorithm (Mirjalili & Lewis, 2016)
  - EOGWO : *proposed* Enhanced Opposition-based Grey Wolf Optimizer

Every optimizer exposes the same interface:
    best_val, best_pos, curve = optimize(f, lb, ub, D, pop, max_iter, rng)
where `curve` is the best-so-far objective value recorded once per iteration.
All evaluations are vectorized over the population where possible.
"""
import numpy as np
from math import gamma as _gamma


def _clip(x, lb, ub):
    return np.clip(x, lb, ub)


# ---------------------------------------------------------------------------
# Grey Wolf Optimizer (canonical)
# ---------------------------------------------------------------------------
def gwo(f, lb, ub, D, pop, max_iter, rng):
    X = rng.uniform(lb, ub, size=(pop, D))
    fit = f(X)
    order = np.argsort(fit)
    alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    fa, fb, fd = fit[order[0]], fit[order[1]], fit[order[2]]
    curve = np.empty(max_iter)
    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter
        for i in range(pop):
            A1 = a * (2 * rng.random(D) - 1); C1 = 2 * rng.random(D)
            A2 = a * (2 * rng.random(D) - 1); C2 = 2 * rng.random(D)
            A3 = a * (2 * rng.random(D) - 1); C3 = 2 * rng.random(D)
            X1 = alpha - A1 * np.abs(C1 * alpha - X[i])
            X2 = beta  - A2 * np.abs(C2 * beta  - X[i])
            X3 = delta - A3 * np.abs(C3 * delta - X[i])
            X[i] = _clip((X1 + X2 + X3) / 3.0, lb, ub)
        fit = f(X)
        for i in range(pop):
            if fit[i] < fa:
                fd, delta = fb, beta.copy(); fb, beta = fa, alpha.copy(); fa, alpha = fit[i], X[i].copy()
            elif fit[i] < fb:
                fd, delta = fb, beta.copy(); fb, beta = fit[i], X[i].copy()
            elif fit[i] < fd:
                fd, delta = fit[i], X[i].copy()
        curve[t] = fa
    return fa, alpha, curve


# ---------------------------------------------------------------------------
# Particle Swarm Optimization
# ---------------------------------------------------------------------------
def pso(f, lb, ub, D, pop, max_iter, rng):
    w_max, w_min, c1, c2 = 0.9, 0.4, 2.0, 2.0
    X = rng.uniform(lb, ub, size=(pop, D))
    vmax = 0.2 * (ub - lb)
    V = rng.uniform(-vmax, vmax, size=(pop, D))
    fit = f(X)
    pbest = X.copy(); pbest_f = fit.copy()
    g = int(np.argmin(fit)); gbest = X[g].copy(); gbest_f = fit[g]
    curve = np.empty(max_iter)
    for t in range(max_iter):
        w = w_max - (w_max - w_min) * t / max_iter
        r1 = rng.random((pop, D)); r2 = rng.random((pop, D))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = _clip(X + V, lb, ub)
        fit = f(X)
        imp = fit < pbest_f
        pbest[imp] = X[imp]; pbest_f[imp] = fit[imp]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest_f = pbest_f[g]; gbest = pbest[g].copy()
        curve[t] = gbest_f
    return gbest_f, gbest, curve


# ---------------------------------------------------------------------------
# Differential Evolution (DE/rand/1/bin)
# ---------------------------------------------------------------------------
def de(f, lb, ub, D, pop, max_iter, rng, F=0.5, CR=0.9):
    X = rng.uniform(lb, ub, size=(pop, D))
    fit = f(X)
    curve = np.empty(max_iter)
    idx_all = np.arange(pop)
    for t in range(max_iter):
        for i in range(pop):
            choices = idx_all[idx_all != i]
            r1, r2, r3 = rng.choice(choices, 3, replace=False)
            mutant = X[r1] + F * (X[r2] - X[r3])
            mutant = _clip(mutant, lb, ub)
            cross = rng.random(D) < CR
            if not np.any(cross):
                cross[rng.integers(D)] = True
            trial = np.where(cross, mutant, X[i])
            ft = f(trial.reshape(1, -1))
            if ft < fit[i]:
                X[i] = trial; fit[i] = ft
        curve[t] = fit.min()
    g = int(np.argmin(fit))
    return fit[g], X[g], curve


# ---------------------------------------------------------------------------
# Whale Optimization Algorithm
# ---------------------------------------------------------------------------
def woa(f, lb, ub, D, pop, max_iter, rng):
    X = rng.uniform(lb, ub, size=(pop, D))
    fit = f(X)
    g = int(np.argmin(fit)); best = X[g].copy(); best_f = fit[g]
    curve = np.empty(max_iter)
    b = 1.0
    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter
        for i in range(pop):
            r = rng.random(); A = 2 * a * r - a; C = 2 * rng.random()
            p = rng.random()
            if p < 0.5:
                if abs(A) < 1:
                    Dvec = np.abs(C * best - X[i]); newx = best - A * Dvec
                else:
                    rand = X[rng.integers(pop)]
                    Dvec = np.abs(C * rand - X[i]); newx = rand - A * Dvec
            else:
                l = rng.uniform(-1, 1, D)
                Dvec = np.abs(best - X[i])
                newx = Dvec * np.exp(b * l) * np.cos(2 * np.pi * l) + best
            X[i] = _clip(newx, lb, ub)
        fit = f(X)
        gi = int(np.argmin(fit))
        if fit[gi] < best_f:
            best_f = fit[gi]; best = X[gi].copy()
        curve[t] = best_f
    return best_f, best, curve


# ---------------------------------------------------------------------------
# PROPOSED: Enhanced Opposition-based Grey Wolf Optimizer (EOGWO)
# ---------------------------------------------------------------------------
# Three coordinated enhancements over canonical GWO:
#   (E1) Nonlinear cosine-controlled convergence parameter a(t), replacing the
#        linear decay. This lengthens the exploration phase early and sharpens
#        exploitation late, addressing GWO's premature-convergence tendency.
#   (E2) Opposition-Based Learning (OBL) at initialization AND a dynamic
#        generalized-OBL re-seeding of the worst sub-population whenever the
#        leader stagnates, expanding effective search coverage.
#   (E3) Levy-flight perturbation of the alpha (leader) every iteration, giving
#        occasional long jumps to escape local optima while preserving the
#        cooperative three-leader guidance of GWO.
# ---------------------------------------------------------------------------
def _levy_step(D, rng, beta=1.5):
    sigma = (_gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, D)
    v = rng.normal(0, 1, D)
    return u / (np.abs(v) ** (1 / beta))


def eogwo(f, lb, ub, D, pop, max_iter, rng,
          stag_window=8, reseed_frac=0.3):
    # (E2) OBL-augmented initialization: generate pop random + opposites, keep best pop
    X0 = rng.uniform(lb, ub, size=(pop, D))
    Xopp = (lb + ub) - X0
    cand = np.vstack([X0, Xopp])
    fc = f(cand)
    keep = np.argsort(fc)[:pop]
    X = cand[keep].copy()
    fit = fc[keep].copy()

    order = np.argsort(fit)
    alpha, beta_, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    fa, fb, fd = fit[order[0]], fit[order[1]], fit[order[2]]
    curve = np.empty(max_iter)
    last_improve = 0
    prev_fa = fa

    for t in range(max_iter):
        # (E1) nonlinear cosine convergence parameter
        a = 2.0 * np.cos((np.pi / 2.0) * (t / max_iter))
        for i in range(pop):
            A1 = a * (2 * rng.random(D) - 1); C1 = 2 * rng.random(D)
            A2 = a * (2 * rng.random(D) - 1); C2 = 2 * rng.random(D)
            A3 = a * (2 * rng.random(D) - 1); C3 = 2 * rng.random(D)
            X1 = alpha  - A1 * np.abs(C1 * alpha  - X[i])
            X2 = beta_  - A2 * np.abs(C2 * beta_  - X[i])
            X3 = delta  - A3 * np.abs(C3 * delta  - X[i])
            X[i] = _clip((X1 + X2 + X3) / 3.0, lb, ub)
        fit = f(X)

        # update leaders
        for i in range(pop):
            if fit[i] < fa:
                fd, delta = fb, beta_.copy(); fb, beta_ = fa, alpha.copy(); fa, alpha = fit[i], X[i].copy()
            elif fit[i] < fb:
                fd, delta = fb, beta_.copy(); fb, beta_ = fit[i], X[i].copy()
            elif fit[i] < fd:
                fd, delta = fit[i], X[i].copy()

        # (E3) Levy-flight perturbation of the leader
        step = 0.01 * _levy_step(D, rng) * (ub - lb)
        cand_alpha = _clip(alpha + step * (rng.random(D) < 0.5), lb, ub)
        fca = f(cand_alpha.reshape(1, -1))
        if fca < fa:
            fa, alpha = float(fca), cand_alpha
            # demote chain
            if fa < fb:
                pass

        # stagnation tracking + (E2) dynamic generalized-OBL re-seeding
        if fa < prev_fa - 1e-12:
            last_improve = t
            prev_fa = fa
        if t - last_improve >= stag_window:
            n_re = max(1, int(reseed_frac * pop))
            worst = np.argsort(fit)[-n_re:]
            # generalized OBL around current search-space dynamic bounds
            dlo, dhi = X.min(axis=0), X.max(axis=0)
            k = rng.uniform(0.5, 1.0)
            X[worst] = _clip(k * (dlo + dhi) - X[worst], lb, ub)
            fit[worst] = f(X[worst])
            last_improve = t  # reset window after intervention
        curve[t] = fa
    return fa, alpha, curve


ALGORITHMS = {
    "EOGWO": eogwo,
    "GWO": gwo,
    "PSO": pso,
    "DE": de,
    "WOA": woa,
}
